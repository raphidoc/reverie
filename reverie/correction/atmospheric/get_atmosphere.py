import os

import numpy as np
import scipy.interpolate as sp
from scipy.interpolate import RegularGridInterpolator
import xarray as xr

from reverie import ReveCube
from reverie.correction.atmospheric import lut

def build_aer_interpolators(aer_lut):
    """
    Build RegularGridInterpolator objects for aer LUT variables used in get_ra.
    Returns dict of interpolators and the grid points tuple.
    """
    # Convert coordinate arrays to numpy (float32) and ensure order matches LUT dims
    sol_zen = np.asarray(aer_lut.sol_zen.values, dtype=np.float32)
    view_zen = np.asarray(aer_lut.view_zen.values, dtype=np.float32)
    rel_az = np.asarray(aer_lut.relative_azimuth.values, dtype=np.float32)
    aot550 = np.asarray(aer_lut.aot550.values, dtype=np.float32)
    target_pressure = np.asarray(aer_lut.target_pressure.values, dtype=np.float32)
    sensor_alt = np.asarray(aer_lut.sensor_altitude.values, dtype=np.float32)
    wavelength = np.asarray(aer_lut.wavelength.values, dtype=np.float32)

    points = (sol_zen, view_zen, rel_az, aot550, target_pressure, sensor_alt, wavelength)

    # Ensure LUT data arrays are contiguous np.float32 in memory
    # variable names used in original code:
    rho_path_vals = np.asarray(aer_lut["atmospheric_reflectance_at_sensor"].values, dtype=np.float32)
    t_ra_vals = np.asarray(aer_lut["total_scattering_trans_total"].values, dtype=np.float32)
    s_ra_vals = np.asarray(aer_lut["spherical_albedo_total"].values, dtype=np.float32)

    # Build interpolators (bounds_error=False to allow extrapolation as nan/fill_value)
    interp_rho = RegularGridInterpolator(points, rho_path_vals,
                                        method='linear', bounds_error=True, fill_value=np.nan)
    interp_t = RegularGridInterpolator(points, t_ra_vals,
                                       method='linear', bounds_error=True, fill_value=np.nan)
    interp_s = RegularGridInterpolator(points, s_ra_vals,
                                       method='linear', bounds_error=True, fill_value=np.nan)

    return dict(rho=interp_rho, t=interp_t, s=interp_s), points

def build_gas_interpolator(gas_lut):
    """
    Build RegularGridInterpolator for gas LUT variable global_gas_trans_total.
    """
    sol_zen = np.asarray(gas_lut.sol_zen.values, dtype=np.float32)
    view_zen = np.asarray(gas_lut.view_zen.values, dtype=np.float32)
    rel_az = np.asarray(gas_lut.relative_azimuth.values, dtype=np.float32)
    water = np.asarray(gas_lut.water.values, dtype=np.float32)
    ozone = np.asarray(gas_lut.ozone.values, dtype=np.float32)
    target_pressure = np.asarray(gas_lut.target_pressure.values, dtype=np.float32)
    sensor_alt = np.asarray(gas_lut.sensor_altitude.values, dtype=np.float32)
    wavelength = np.asarray(gas_lut.wavelength.values, dtype=np.float32)

    points = (sol_zen, view_zen, rel_az, water, ozone, target_pressure, sensor_alt, wavelength)
    t_gas_vals = np.asarray(gas_lut["global_gas_trans_total"].values, dtype=np.float32)

    interp_tgas = RegularGridInterpolator(points, t_gas_vals,
                                         method='linear', bounds_error=True, fill_value=np.nan)
    return interp_tgas, points

def get_ra(image: ReveCube, window, wavelength, aod555, interp_rho, interp_t, interp_s):

    # aer_lut = lut.load_aer()

    if window is not None:
        x = image.x[window.col_off: window.col_off + window.width]
        y = image.y[window.row_off: window.row_off + window.height]
        y_slice = slice(int(window.row_off), int(window.row_off + window.height))
        x_slice = slice(int(window.col_off), int(window.col_off + window.width))
    else:
        x = image.x.values
        y = image.y.values
        x_slice = slice(None)
        y_slice = slice(None)

    ra_components = xr.Dataset(
        data_vars=dict(
            rho_path=(
                ["wavelength", "y", "x"],
                np.full((len(wavelength), len(y), len(x)), np.nan, np.float32),
            ),
            trans_ra=(
                ["wavelength", "y", "x"],
                np.full((len(wavelength), len(y), len(x)), np.nan, np.float32),
            ),
            spherical_albedo_ra=(
                ["wavelength", "y", "x"],
                np.full((len(wavelength), len(y), len(x)), np.nan, np.float32),
            ),
        ),
        coords=dict(
            wavelength=wavelength,
            y=("y", y),
            x=("x", x),
        ),
    )
    # create placeholder for (x, y) gas_trans
    window_shape = image.get_valid_mask(window=window).shape
    # window_shape = (len(wavelength), len(y), len(x)) #

    rho_path_ra = np.full(window_shape, np.nan, np.float32)
    t_ra = np.full(window_shape, np.nan, np.float32)
    s_ra = np.full(window_shape, np.nan, np.float32)

    # flat_rho_path_ra = rho_path_ra.reshape(len(wavelength), -1)
    # flat_t_ra = t_ra.reshape(len(wavelength), -1)
    # flat_s_ra = s_ra.reshape(len(wavelength), -1)

    # Do gas correction with 6S output "global_gas_trans_total"
    # It integrates all gas transmittance in the downward and upward direction
    # Dimensions of the LUT are {wavelength, sun_zenith, view_zenith, relative_azimuth,  sensor_altitude, water_vapor, ozone, target_pressure}
    # Variables taken from the image {wavelength, sun_zenith, view_zenith, relative_azimuth, sensor_altitude}
    # Variables taken from the GMAO MERRA2 model {water_vapor, ozone, target_pressure}

    # aer_points = (
    #     aer_lut.sol_zen.values,
    #     aer_lut.view_zen.values,
    #     aer_lut.relative_azimuth.values,
    #     aer_lut.aot550.values,
    #     aer_lut.target_pressure.values,
    #     aer_lut.sensor_altitude.values,
    #     aer_lut.wavelength.values,
    # )

    valid_mask = image.get_valid_mask(window)
    flat_valid_mask = valid_mask.flatten()

    n_pixels = len(
        image.in_ds.isel(x=x_slice, y=y_slice)["sun_zenith"]
        .values[valid_mask]
        .reshape(-1, 1)
    )

    if aod555 is not None:
        aod555_xi = aod555
    else:
        aod555_xi = image.variables[
                        "aerosol_optical_thickness_at_555_nm"
                    ].values

    sun_zenith_xi = image.in_ds.isel(x=x_slice, y=y_slice)["sun_zenith"].values[valid_mask]
    view_zenith_xi = image.in_ds.isel(x=x_slice, y=y_slice)["view_zenith"].values[valid_mask]
    relative_azimuth_xi = image.in_ds.isel(x=x_slice, y=y_slice)["relative_azimuth"].values[valid_mask]
    traget_pressure_xi = image.in_ds["surface_air_pressure"].values
    sensor_altitude_xi = image.z.values

    # n_wavelengths = len(wavelength)
    #
    # sun_zenith_col = np.repeat(sun_zenith_xi, n_wavelengths).reshape(-1, 1)
    # view_zenith_col = np.repeat(view_zenith_xi, n_wavelengths).reshape(-1, 1)
    # relative_azimuth_col = np.repeat(relative_azimuth_xi, n_wavelengths).reshape(-1, 1)
    # aod555_col = np.repeat(aod555_xi, n_pixels * n_wavelengths).reshape(-1, 1)
    # traget_pressure_col = np.repeat(traget_pressure_xi, n_pixels * n_wavelengths).reshape(-1, 1)
    # sensor_altitude_col = np.repeat(sensor_altitude_xi, n_pixels * n_wavelengths).reshape(-1, 1)
    # wavelength_col = np.tile(wavelength, n_pixels).reshape(-1, 1)
    #
    # aer_xi = np.hstack([
    #     sun_zenith_col,
    #     view_zenith_col,
    #     relative_azimuth_col,
    #     aod555_col,
    #     traget_pressure_col,
    #     sensor_altitude_col,
    #     wavelength_col,
    # ])

    for i, wl in enumerate(wavelength):
        aer_xi = np.hstack(
            [
                sun_zenith_xi.reshape(-1, 1),
                view_zenith_xi.reshape(-1, 1),
                relative_azimuth_xi.reshape(-1, 1),
                np.repeat(aod555_xi,n_pixels,).reshape(-1, 1),
                np.repeat(traget_pressure_xi, n_pixels).reshape(-1, 1),
                np.repeat(sensor_altitude_xi, n_pixels).reshape(-1, 1),
                np.repeat(wl, n_pixels).reshape(-1, 1),
            ]
        )

        # rho_path_ra_values = sp.interpn(
        #     points=aer_points,
        #     # values=total_lut["atmospheric_reflectance_at_sensor"][
        #     values=aer_lut["atmospheric_reflectance_at_sensor"][:, :, :, :, :, :, :].values,
        #     xi=aer_xi,
        # )
        rho_path_ra_values = interp_rho(aer_xi)
        rho_path_ra[valid_mask] = rho_path_ra_values
        ra_components["rho_path"][i, :, :] = rho_path_ra

        # t_ra_values = sp.interpn(
        #     points=aer_points,
        #     values=aer_lut["total_scattering_trans_total"][
        #            :, :, :, :, :, :, :
        #            ].values,
        #     xi=aer_xi,
        # )
        t_ra_values = interp_t(aer_xi)
        t_ra[valid_mask] = t_ra_values
        ra_components["trans_ra"][i, :, :] = t_ra

        # s_ra_values = sp.interpn(
        #     points=aer_points,
        #     values=aer_lut["spherical_albedo_total"][
        #            :, :, :, :, :, :, :
        #            ].values,
        #     xi=aer_xi,
        # )
        s_ra_values = interp_s(aer_xi)
        s_ra[valid_mask] = s_ra_values
        ra_components["spherical_albedo_ra"][i, :, :] = s_ra

    # t_ra_values = interp_t(aer_xi)
    # t_ra_values = t_ra_values.reshape(len(wavelength), flat_valid_mask.sum())
    # flat_t_ra[:, flat_valid_mask] = t_ra_values
    # t_ra = flat_t_ra.reshape(t_ra.shape)
    #
    # rho_path_ra_values = interp_rho(aer_xi)
    # for i in range(n_wavelengths):
    #     rho_path_ra[i, valid_mask] = rho_path_ra_values[i]
    # rho_path_ra_values = rho_path_ra_values.reshape(len(wavelength), flat_valid_mask.sum())
    # flat_rho_path_ra[:, flat_valid_mask] = rho_path_ra_values
    # rho_path_ra = flat_rho_path_ra.reshape(rho_path_ra.shape)
    #
    # s_ra_values = interp_s(aer_xi)
    # s_ra_values = s_ra_values.reshape(len(wavelength), flat_valid_mask.sum())
    # flat_s_ra[:, flat_valid_mask] = s_ra_values
    # s_ra = flat_s_ra.reshape(s_ra.shape)

    # return rho_path_ra, t_ra, s_ra
    return ra_components

def get_gas(image: ReveCube, window, wavelength, interp_tgas):

    gas_lut = lut.load_gas()

    if window is not None:
        x = image.x[window.col_off: window.col_off + window.width]
        y = image.y[window.row_off: window.row_off + window.height]
        y_slice = slice(int(window.row_off), int(window.row_off + window.height))
        x_slice = slice(int(window.col_off), int(window.col_off + window.width))
    else:
        x = image.x.values
        y = image.y.values
        x_slice = slice(None)
        y_slice = slice(None)

    gas_components = xr.Dataset(
        data_vars=dict(
            t_gas=(
                ["wavelength", "y", "x"],
                np.full((len(wavelength), len(y), len(x)), np.nan, np.float32),
            ),
        ),
        coords=dict(
            wavelength=wavelength,
            y=("y", y),
            x=("x", x),
        ),
    )

    # create placeholder for (x, y) gas_trans
    window_shape = image.get_valid_mask(window=window).shape
    # window_shape = (len(wavelength), len(y), len(x))

    t_gas = np.full(window_shape, np.nan, np.float32)
    # flat_t_gas = t_gas.reshape(len(wavelength), -1)

    # Do gas correction with 6S output "global_gas_trans_total"
    # It integrates all gas transmittance in the downward and upward direction
    # Dimensions of the LUT are {wavelength, sun_zenith, view_zenith, relative_azimuth,  sensor_altitude, water_vapor, ozone, target_pressure}
    # Variables taken from the image {wavelength, sun_zenith, view_zenith, relative_azimuth, sensor_altitude}
    # Variables taken from the GMAO MERRA2 model {water_vapor, ozone, target_pressure}

    # gas_points = (
    #     gas_lut.sol_zen.values,
    #     gas_lut.view_zen.values,
    #     gas_lut.relative_azimuth.values,
    #     gas_lut.water.values,
    #     gas_lut.ozone.values,
    #     gas_lut.target_pressure.values,
    #     gas_lut.sensor_altitude.values,
    #     gas_lut.wavelength.values,
    # )

    valid_mask = image.get_valid_mask(window)
    flat_valid_mask = valid_mask.flatten()

    n_pixels = len(
        image.in_ds.isel(x=x_slice, y=y_slice)["sun_zenith"]
        .values[valid_mask]
        .reshape(-1, 1)
    )

    sun_zenith_xi = image.in_ds.isel(x=x_slice, y=y_slice)["sun_zenith"].values[valid_mask]
    view_zenith_xi = image.in_ds.isel(x=x_slice, y=y_slice)["view_zenith"].values[valid_mask]
    relative_azimuth_xi = image.in_ds.isel(x=x_slice, y=y_slice)["relative_azimuth"].values[valid_mask]
    water_vapor_xi = image.in_ds["atmosphere_mass_content_of_water_vapor"].values
    ozone_xi = image.in_ds["equivalent_thickness_at_stp_of_atmosphere_ozone_content"].values
    traget_pressure_xi = image.in_ds["surface_air_pressure"].values
    sensor_altitude_xi = image.z.values

    # n_wavelengths = len(wavelength)
    #
    # sun_zenith_col = np.repeat(sun_zenith_xi, n_wavelengths).reshape(-1, 1)
    # view_zenith_col = np.repeat(view_zenith_xi, n_wavelengths).reshape(-1, 1)
    # relative_azimuth_col = np.repeat(relative_azimuth_xi, n_wavelengths).reshape(-1, 1)
    # water_vapor_col = np.repeat(water_vapor_xi, n_pixels * n_wavelengths).reshape(-1, 1)
    # ozone_col = np.repeat(ozone_xi, n_pixels * n_wavelengths).reshape(-1, 1)
    # traget_pressure_col = np.repeat(traget_pressure_xi, n_pixels * n_wavelengths).reshape(-1, 1)
    # sensor_altitude_col = np.repeat(sensor_altitude_xi, n_pixels * n_wavelengths).reshape(-1, 1)
    # wavelength_col = np.tile(wavelength, n_pixels).reshape(-1, 1)
    #
    # gas_xi = np.hstack([
    #     sun_zenith_col,
    #     view_zenith_col,
    #     relative_azimuth_col,
    #     water_vapor_col,
    #     ozone_col,
    #     traget_pressure_col,
    #     sensor_altitude_col,
    #     wavelength_col,
    # ])

    for i, wl in enumerate(wavelength):
        gas_xi = np.hstack(
            [
                sun_zenith_xi.reshape(-1, 1),
                view_zenith_xi.reshape(-1, 1),
                relative_azimuth_xi.reshape(-1, 1),
                np.repeat(water_vapor_xi, n_pixels).reshape(-1, 1),
                np.repeat(ozone_xi, n_pixels).reshape(-1, 1),
                np.repeat(traget_pressure_xi, n_pixels).reshape(-1, 1),
                np.repeat(sensor_altitude_xi, n_pixels).reshape(-1, 1),
                np.repeat(wl, n_pixels).reshape(-1, 1),
            ]
        )

        # t_gas_values = sp.interpn(
        #     points=gas_points,
        #     # values=total_lut["atmospheric_reflectance_at_sensor"][
        #     values=gas_lut["global_gas_trans_total"][:, :, :, :, :, :, :, :].values,
        #     xi=gas_xi,
        # )
        t_gas_values = interp_tgas(gas_xi)
        t_gas[valid_mask] = t_gas_values
        gas_components["t_gas"][i, :, :] = t_gas



    # t_gas_values = interp_tgas(gas_xi)
    # t_gas_values = t_gas_values.reshape(len(wavelength), flat_valid_mask.sum())
    # flat_t_gas[:, flat_valid_mask] = t_gas_values
    # t_gas = flat_t_gas.reshape(t_gas.shape)

    return gas_components

def get_ra_vectorized(image: ReveCube, window, wavelength, aod555, interp_rho, interp_t, interp_s):
    """
    Vectorized get_ra: returns numpy arrays (rho_path_ra, t_ra, s_ra)
    shape = (n_wavelengths, ny, nx) as float32
    """
    if window is not None:
        x = image.x[window.col_off: window.col_off + window.width]
        y = image.y[window.row_off: window.row_off + window.height]
        y_slice = slice(int(window.row_off), int(window.row_off + window.height))
        x_slice = slice(int(window.col_off), int(window.col_off + window.width))
    else:
        x = image.x
        y = image.y
        x_slice = slice(None)
        y_slice = slice(None)

    valid_mask = image.get_valid_mask(window)  # boolean (ny, nx)
    flat_valid = valid_mask.flatten()
    n_pixels = flat_valid.sum()
    if n_pixels == 0:
        # return empty arrays shaped correctly
        ny = len(y); nx = len(x); n_wl = len(wavelength)
        shape = (n_wl, ny, nx)
        return (np.full(shape, np.nan, np.float32),
                np.full(shape, np.nan, np.float32),
                np.full(shape, np.nan, np.float32))

    # coords for valid pixels
    sun_zenith = image.in_ds.isel(x=x_slice, y=y_slice)["sun_zenith"].values[valid_mask].astype(np.float32)
    view_zenith = image.in_ds.isel(x=x_slice, y=y_slice)["view_zenith"].values[valid_mask].astype(np.float32)
    rel_az = image.in_ds.isel(x=x_slice, y=y_slice)["relative_azimuth"].values[valid_mask].astype(np.float32)

    # scalar image parameters
    target_pressure = np.asarray(image.in_ds["surface_air_pressure"].values, dtype=np.float32)
    sensor_altitude = np.asarray(image.z, dtype=np.float32)

    # aerosol optical thickness: scalar or image provided
    if aod555 is not None:
        aod_val = float(aod555)
        aod_arr = np.full(n_pixels, aod_val, dtype=np.float32)
    else:
        # if image variable, broadcast per pixel (same scalar per pixel in your code)
        aod_arr = np.repeat(image.in_ds["aerosol_optical_thickness_at_555_nm"].values, n_pixels).astype(np.float32)

    n_wl = len(wavelength)

    # Build full xi for all valid pixels × wavelengths.
    # We'll create arrays of shape (n_pixels * n_wavelengths, ndim)
    # Dimensions for aer_xi: [sun_zenith, view_zenith, rel_az, aot550, target_pressure, sensor_altitude, wavelength]
    # We'll stack in order expected by the interpolators.

    # create per-pixel columns, then tile for wavelengths
    sun_col = np.repeat(sun_zenith, n_wl).reshape(-1, 1)
    view_col = np.repeat(view_zenith, n_wl).reshape(-1, 1)
    relcol = np.repeat(rel_az, n_wl).reshape(-1, 1)
    aod_col = np.repeat(aod_arr, n_wl).reshape(-1, 1)
    press_col = np.repeat(target_pressure, n_pixels * n_wl).reshape(-1, 1)
    sensor_col = np.repeat(sensor_altitude,n_pixels * n_wl).reshape(-1, 1)

    # wavelength varies fastest: tile wavelengths for each pixel
    wl_col = np.tile(wavelength.astype(np.float32), n_pixels).reshape(-1, 1)

    aer_xi = np.hstack([sun_col, view_col, relcol, aod_col, press_col, sensor_col, wl_col])

    # Perform vectorized interpolation (single call each)
    # Output arrays are length (n_pixels * n_wl)
    rho_vals = interp_rho(aer_xi)
    t_vals = interp_t(aer_xi)
    s_vals = interp_s(aer_xi)

    # reshape back to (n_wl, n_pixels)
    rho_vals = rho_vals.reshape(n_pixels, n_wl).T  # (n_wl, n_pixels)
    t_vals = t_vals.reshape(n_pixels, n_wl).T
    s_vals = s_vals.reshape(n_pixels, n_wl).T

    # Build full arrays (wavelength, ny, nx) filled with nan and insert valid pixels
    ny = len(y); nx = len(x)
    rho_full = np.full((n_wl, ny * nx), np.nan, dtype=np.float32)
    t_full = np.full_like(rho_full, np.nan, dtype=np.float32)
    s_full = np.full_like(rho_full, np.nan, dtype=np.float32)

    rho_full[:, flat_valid] = rho_vals
    t_full[:, flat_valid] = t_vals
    s_full[:, flat_valid] = s_vals

    rho_full = rho_full.reshape(n_wl, ny, nx)
    t_full = t_full.reshape(n_wl, ny, nx)
    s_full = s_full.reshape(n_wl, ny, nx)

    return rho_full, t_full, s_full

def get_gas_vectorized(image: ReveCube, window, wavelength, interp_tgas):
    if window is not None:
        x = image.x[window.col_off: window.col_off + window.width]
        y = image.y[window.row_off: window.row_off + window.height]
        y_slice = slice(int(window.row_off), int(window.row_off + window.height))
        x_slice = slice(int(window.col_off), int(window.col_off + window.width))
    else:
        x = image.x
        y = image.y
        x_slice = slice(None)
        y_slice = slice(None)

    valid_mask = image.get_valid_mask(window)
    flat_valid = valid_mask.flatten()
    n_pixels = flat_valid.sum()
    if n_pixels == 0:
        ny = len(y); nx = len(x); n_wl = len(wavelength)
        return np.full((n_wl, ny, nx), np.nan, np.float32)

    sun_zenith = image.in_ds.isel(x=x_slice, y=y_slice)["sun_zenith"].values[valid_mask].astype(np.float32)
    view_zenith = image.in_ds.isel(x=x_slice, y=y_slice)["view_zenith"].values[valid_mask].astype(np.float32)
    rel_az = image.in_ds.isel(x=x_slice, y=y_slice)["relative_azimuth"].values[valid_mask].astype(np.float32)

    water = np.asarray(image.in_ds["atmosphere_mass_content_of_water_vapor"].values, dtype=np.float32)
    ozone = np.asarray(image.in_ds["equivalent_thickness_at_stp_of_atmosphere_ozone_content"].values, dtype=np.float32)
    target_pressure = np.asarray(image.in_ds["surface_air_pressure"].values, dtype=np.float32)
    sensor_altitude = np.asarray(image.z, dtype=np.float32)

    n_wl = len(wavelength)
    sun_col = np.repeat(sun_zenith, n_wl).reshape(-1, 1)
    view_col = np.repeat(view_zenith, n_wl).reshape(-1, 1)
    rel_col = np.repeat(rel_az, n_wl).reshape(-1, 1)
    water_col = np.repeat(water, n_pixels * n_wl).reshape(-1, 1)   # scalar usually
    ozone_col = np.repeat(ozone, n_pixels * n_wl).reshape(-1, 1)
    press_col = np.repeat(target_pressure, n_pixels * n_wl).reshape(-1, 1)
    sensor_col = np.repeat(sensor_altitude, n_pixels * n_wl).reshape(-1, 1)
    wl_col = np.tile(wavelength.astype(np.float32), n_pixels).reshape(-1, 1)

    gas_xi = np.hstack([sun_col, view_col, rel_col, water_col, ozone_col, press_col, sensor_col, wl_col])

    tgas_vals = interp_tgas(gas_xi)
    tgas_vals = tgas_vals.reshape(n_pixels, n_wl).T  # (n_wl, n_pixels)

    ny = len(y); nx = len(x)
    tgas_full = np.full((n_wl, ny * nx), np.nan, dtype=np.float32)
    tgas_full[:, flat_valid] = tgas_vals
    tgas_full = tgas_full.reshape(n_wl, ny, nx)
    return tgas_full

def get_gas_ra_combined(image: ReveCube, window, wavelength):

    # wavelength = image.wavelength

    x = image.x.values
    y = image.y.values

    atmosphere_components = xr.Dataset(
        data_vars=dict(
            atmospheric_path_reflectance=(
                ["wavelength", "y", "x"],
                np.full((len(wavelength), len(y), len(x)), np.nan, float),
            ),
            gas_path_reflectance=(
                ["wavelength", "y", "x"],
                np.full((len(wavelength), len(y), len(x)), np.nan, float),
            ),
            gas_trans_total=(
                ["wavelength", "y", "x"],
                np.full((len(wavelength), len(y), len(x)), np.nan, float),
            ),
            rayleigh_trans_total=(
                ["wavelength", "y", "x"],
                np.full((len(wavelength), len(y), len(x)), np.nan, float),
            ),
            spherical_albedo_rayleigh=(
                ["wavelength", "y", "x"],
                np.full((len(wavelength), len(y), len(x)), np.nan, float),
            ),
            aer_path_reflectance=(
                ["wavelength", "y", "x"],
                np.full((len(wavelength), len(y), len(x)), np.nan, float),
            ),
            aerosol_trans_total=(
                ["wavelength", "y", "x"],
                np.full((len(wavelength), len(y), len(x)), np.nan, float),
            ),
            spherical_albedo_aerosol=(
                ["wavelength", "y", "x"],
                np.full((len(wavelength), len(y), len(x)), np.nan, float),
            ),
            spherical_albedo_total=(
                ["wavelength", "y", "x"],
                np.full((len(wavelength), len(y), len(x)), np.nan, float),
            ),
        ),
        coords=dict(
            wavelength=wavelength,
            y=("y", y),
            x=("x", x),
        ),
    )

    window_shape = image.get_valid_mask(window=window).shape

    # create placeholder for (x, y) gas_trans
    pr = np.full(window_shape, np.nan, float)
    gt = np.full(window_shape, np.nan, float)
    rt = np.full(window_shape, np.nan, float)
    at = np.full(window_shape, np.nan, float)
    tot_sa = np.full(window_shape, np.nan, float)

    # Do gas correction with 6S output "global_gas_trans_total"
    # It integrates all gas transmittance in the downward and upward direction
    # Dimensions of the LUT are {wavelength, sun_zenith, view_zenith, relative_azimuth,  sensor_altitude, water_vapor, ozone, target_pressure}
    # Variables taken from the image {wavelength, sun_zenith, view_zenith, relative_azimuth, sensor_altitude}
    # Variables taken from the GMAO MERRA2 model {water_vapor, ozone, target_pressure}

    lut = lut.load_aer()

    lut_points = (
        lut.sol_zen.values,
        lut.view_zen.values,
        lut.relative_azimuth.values,
        lut.water.values,
        lut.ozone.values,
        lut.aot550.values,
        lut.target_pressure.values,
        lut.sensor_altitude.values,
        lut.wavelength.values,
    )

    n_pixels = len(
        image.variables["sun_zenith"]
        .values[image.get_valid_mask(window=window)]
        .reshape(-1, 1)
    )

    for index, band in enumerate(wavelength):
        xi = np.hstack(
            [
                image.variables["sun_zenith"]
                .values[image.get_valid_mask(window)]
                .reshape(-1, 1),
                image.variables["view_zenith"]
                .values[image.get_valid_mask(window)]
                .reshape(-1, 1),
                image.variables["relative_azimuth"]
                .values[image.get_valid_mask(window)]
                .reshape(-1, 1),
                np.repeat(
                    image.variables[
                        "atmosphere_mass_content_of_water_vapor"
                    ].values,
                    n_pixels,
                ).reshape(-1, 1),
                np.repeat(
                    image.variables[
                        "equivalent_thickness_at_stp_of_atmosphere_ozone_content"
                    ].values,
                    n_pixels,
                ).reshape(-1, 1),
                np.repeat(
                    image.variables[
                        "aerosol_optical_thickness_at_555_nm"
                    ].values,
                    n_pixels,
                ).reshape(-1, 1),
                np.repeat(
                    image.variables["surface_air_pressure"].values, n_pixels
                ).reshape(-1, 1),
                np.repeat((-image.z.values / 1000), n_pixels).reshape(-1, 1),
                np.repeat(band * 1e-3, n_pixels).reshape(-1, 1),
            ]
        )

        pr_values = sp.interpn(
            points=lut_points,
            # values=total_lut["atmospheric_reflectance_at_sensor"][
            values=lut["atmospheric_reflectance_at_sensor"][:, :, :, :, :, :, :, :, : ].values,
            xi=xi,
        )
        pr[image.get_valid_mask(window)] = pr_values
        atmosphere_components["atmospheric_path_reflectance"][index, :, :] = pr

        gt_values = sp.interpn(
            points=lut_points,
            values=lut["global_gas_trans_total"][
                   :, :, :, :, :, :, :, :, :
                   ].values,
            xi=xi,
        )
        gt[image.get_valid_mask(window)] = gt_values
        atmosphere_components["gas_trans_total"][index, :, :] = gt

        rt_values = sp.interpn(
            points=lut_points,
            values=lut["rayleigh_trans_total"][
                   :, :, :, :, :, :, :, :, :
                   ].values,
            xi=xi,
        )
        rt[image.get_valid_mask(window)] = rt_values
        atmosphere_components["rayleigh_trans_total"][index, :, :] = rt

        at_values = sp.interpn(
            points=lut_points,
            values=lut["aerosol_trans_total"][
                   :, :, :, :, :, :, :, :, :
                   ].values,
            xi=xi,
        )
        at[image.get_valid_mask(window)] = at_values
        atmosphere_components["aerosol_trans_total"][index, :, :] = at

        tot_sa_values = sp.interpn(
            points=lut_points,
            values=lut["spherical_albedo_total"][
                   :, :, :, :, :, :, :, :, :
                   ].values,
            xi=xi,
        )
        tot_sa[image.get_valid_mask(window)] = tot_sa_values
        atmosphere_components["spherical_albedo_total"][index, :, :] = tot_sa


    return atmosphere_components