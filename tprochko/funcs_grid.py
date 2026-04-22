# Travis Prochko; from private grid functions compilation
# Grid approximations and spatial derivatives for use 
# on observational or regridded datasets
import numpy as np
import xarray as xr
from scipy import signal
from scipy.ndimage import convolve1d
import cftime

def calc_coriolis(lat):
    """
    Computes coriolis parameter as a function of latitude
    """
    
    omega = 7.2921e-5
    coriolis = 2*omega*np.sin(np.pi*lat/180)

    return coriolis

def earth_radius_xr(lat=None, unit='m'):
    """
    Computes Earth's radius. Returns a scalar (nominal) or latitude-dependent
    radius based on the WGS-84 ellipsoid.

    Parameters
    ----------
    lat : float, ndarray, or xarray.DataArray, optional
        Latitude(s) in degrees. If provided, returns radius as a function of latitude.
    unit : str, optional
        'm' (default) for meters or 'km' for kilometers.

    Returns
    -------
    r : float, ndarray, or xarray.DataArray
        Earth's radius (constant or latitude-dependent), in specified units.
    """

    # WGS-84 ellipsoid radii (in meters)
    a = 6378137.0     # Equatorial radius
    b = 6356752.0     # Polar radius

    # Use nominal mean Earth radius if no latitude provided
    if lat is None:
        r = 6371000.0
    else:
        # Use numpy functions for latitude-dependent radius
        lat_rad = np.radians(lat)
        cos_lat = np.cos(lat_rad)
        sin_lat = np.sin(lat_rad)
        numerator = (a**2 * cos_lat)**2 + (b**2 * sin_lat)**2
        denominator = (a * cos_lat)**2 + (b * sin_lat)**2
        r = np.sqrt(numerator / denominator)

    # Convert to kilometers if requested
    if unit.lower().startswith('km'):
        r = r / 1000.0

    # Return same type as input
    if isinstance(lat, xr.DataArray):
        r = xr.DataArray(r, coords=lat.coords, dims=lat.dims, name='earth_radius')

    return r

def cdtdim_xr(lat, lon, unit='m'):
    """
    Computes approximate zonal (dx) and meridional (dy) dimensions of grid cells
    defined by latitude and longitude 2D arrays.

    Parameters:
    -----------
    lat : xarray.DataArray
        2D array of latitude values (degrees).
    lon : xarray.DataArray
        2D array of longitude values (degrees).
    unit : str, optional
        'm' for meters (default), or 'km' for kilometers.

    Returns:
    --------
    dx : xarray.DataArray
        Zonal width of each grid cell in meters or kilometers.
    dy : xarray.DataArray
        Meridional height of each grid cell in meters or kilometers.
    """
    lat_coords, lon_coords = lat.data, lon.data
    nlat,nlon = len(lat),len(lon)
    # --- Input checks ---
    lat,lon = np.meshgrid(lat,lon)
    if lat.shape != lon.shape:
        raise ValueError("lat and lon must have the same shape.")
    if lat.ndim != 2 or lon.ndim != 2:
        raise ValueError("lat and lon must be 2D arrays.")
    if not (np.all((lat >= -90) & (lat <= 90)) and np.all((lon >= -180) & (lon <= 360))):
        raise ValueError("Some values in lat or lon fall outside valid geographic bounds.")

    # --- Earth's radius (WGS84 average radius) ---
    R = earth_radius_xr(lat) # in meters
    if unit.lower().startswith('km'):
        R /= 1000.0  # convert to kilometers

    # --- Compute gradients ---
    dlat1, dlat2 = np.gradient(lat)
    dlon1, dlon2 = np.gradient(lon)

    # --- Determine axis alignment ---
    if np.allclose(dlat1, 0):
        dlat = dlat2
        dlon = dlon1
    else:
        dlat = dlat1
        dlon = dlon2

    # --- Compute cell dimensions ---
    dy = dlat * R * np.pi / 180.0
    dx = dlon * np.pi / 180.0 * R * np.cos(np.radians(lat))

    # Return as xarray DataArrays (with coordinates and dimensions preserved)

    dx_da = xr.DataArray(dx[0], coords={'lat':lat_coords}, dims=['lat'], name='dx')
    dy_da = xr.DataArray(dy[0], coords={'lat':lat_coords}, dims=['lat'], name='dy')

    return dx_da, dy_da

def xr_first_derivative(da,delta,dim,mask=None,interpolate_na=False,wrap=False):
    """
    Approximates first derivative in a single spatial direction using a central
    finite difference method. Forward and backward finite differences are used
    for boundaries, with the option to wrap longitudes for correct zonal treatment.

    Parameters:
    -----------
    da : xarray.DataArray
        2D (or more) array of values to be differentiated
    delta : xarray.DataArray
        2D array of point-to-point differences in differentiation direction
    dim : str, 'lat' or 'lon'
        Flag indicating direction of differentiation
    mask : xarray.DataArray
        2D array masking out land
    interpolate_na : boolean
        Indicates whether or not to fill np.nan values with neighboring 
        cell values via xarray.DataArray.interpolate_na()
    wrap : boolean
        Indicates whether or not to wrap the longitudes

    Returns:
    --------
    dx : xarray.DataArray
        Zonal width of each grid cell in meters or kilometers.
    dy : xarray.DataArray
        Meridional height of each grid cell in meters or kilometers.
    """

    if interpolate_na == True:
        da = da.interpolate_na(dim=dim,method='linear',fill_value='extrapolate')

    if isinstance(mask, xr.DataArray):
        mask = mask
    if isinstance(mask, str) and mask == 'auto':
        if 'time' in da.dims:
            mask = xr.where(da.isel(time=0,drop=True).notnull(),1.0,np.nan)
        elif 'time' not in da.dims:
            mask = xr.where(da.notnull(),1.0,np.nan)
    elif mask is None:
        mask = 1.0

    if dim == 'lon' and wrap == True:
        # pad in longitude for cyclic wrap
        pad = 1
        da_pad = xr.concat(
            [da.isel(lon=slice(-pad, None)),
             da,
             da.isel(lon=slice(0, pad))],
            dim='lon'
        ).chunk({'lat': -1,'lon': -1})        
    
        # center (central finite difference)
        deriv_center = (da_pad.shift({dim: -1})-da_pad.shift({dim: +1}))/(2*delta)
        # deriv_center = deriv_center.isel({dim:slice(1,-1)})

        deriv = deriv_center.isel(lon=slice(pad,-pad))
        
    else:
         # center (central finite difference)
        deriv_center = (da.shift({dim: -1})-da.shift({dim: +1}))/(2*delta)
        deriv_center = deriv_center.isel({dim:slice(1,-1)})    
        # front boundary (forward finite difference)
        da0 = da.isel({dim:slice(None,2)})
        if dim == 'lat':
            delta0 = delta.isel({dim:slice(None,2)})
        else:
            delta0 = delta
        deriv_front = (da0.shift({dim: -1})-da0)/(delta0)
        deriv_front = deriv_front.isel({dim:slice(None,1)})
        
        # end boundary (backward finite difference)
        da0 = da.isel({dim:slice(-2,None)})
        if dim == 'lat':
            delta0 = delta.isel({dim:slice(-2,None)})
        else:
            delta0 = delta
        deriv_end = (da0.shift({dim: 0})-da0.shift({dim: +1}))/(delta0)
        deriv_end = deriv_end.isel({dim:slice(-1,None)})
    
        deriv = xr.concat([deriv_front,deriv_center,deriv_end],dim=dim)

    return deriv*mask

