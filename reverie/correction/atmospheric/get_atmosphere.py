import os

import numpy as np
import scipy.interpolate as sp
import xarray as xr

from reverie import ReveCube

def get_atmosphere(l1: ReveCube, window, wavelength, image_sub):

    x = image_sub.x.values
    y = image_sub.y.values

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

    window_shape = l1.get_valid_mask(window=window).shape

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

    lut = xr.open_dataset(
        "/home/raphael/PycharmProjects/reverie/reverie/data/lut/lut_combined.nc"
    )

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
        image_sub.variables["sun_zenith"]
        .values[l1.get_valid_mask(window=window)]
        .reshape(-1, 1)
    )

    for index, band in enumerate(wavelength):
        xi = np.hstack(
            [
                image_sub.variables["sun_zenith"]
                .values[l1.get_valid_mask(window)]
                .reshape(-1, 1),
                image_sub.variables["view_zenith"]
                .values[l1.get_valid_mask(window)]
                .reshape(-1, 1),
                image_sub.variables["relative_azimuth"]
                .values[l1.get_valid_mask(window)]
                .reshape(-1, 1),
                np.repeat(
                    image_sub.variables[
                        "atmosphere_mass_content_of_water_vapor"
                    ].values,
                    n_pixels,
                ).reshape(-1, 1),
                np.repeat(
                    image_sub.variables[
                        "equivalent_thickness_at_stp_of_atmosphere_ozone_content"
                    ].values,
                    n_pixels,
                ).reshape(-1, 1),
                np.repeat(
                    image_sub.variables[
                        "aerosol_optical_thickness_at_555_nm"
                    ].values,
                    n_pixels,
                ).reshape(-1, 1),
                np.repeat(
                    image_sub.variables["surface_air_pressure"].values, n_pixels
                ).reshape(-1, 1),
                np.repeat((-image_sub.z.values / 1000), n_pixels).reshape(-1, 1),
                np.repeat(band * 1e-3, n_pixels).reshape(-1, 1),
            ]
        )

        pr_values = sp.interpn(
            points=lut_points,
            # values=total_lut["atmospheric_reflectance_at_sensor"][
            values=lut["atmospheric_reflectance_at_sensor"][:, :, :, :, :, :, :, :, : ].values,
            xi=xi,
        )
        pr[l1.get_valid_mask(window)] = pr_values
        atmosphere_components["atmospheric_path_reflectance"][index, :, :] = pr

        gt_values = sp.interpn(
            points=lut_points,
            values=lut["global_gas_trans_total"][
                   :, :, :, :, :, :, :, :, :
                   ].values,
            xi=xi,
        )
        gt[l1.get_valid_mask(window)] = gt_values
        atmosphere_components["gas_trans_total"][index, :, :] = gt

        rt_values = sp.interpn(
            points=lut_points,
            values=lut["rayleigh_trans_total"][
                   :, :, :, :, :, :, :, :, :
                   ].values,
            xi=xi,
        )
        rt[l1.get_valid_mask(window)] = rt_values
        atmosphere_components["rayleigh_trans_total"][index, :, :] = rt

        at_values = sp.interpn(
            points=lut_points,
            values=lut["aerosol_trans_total"][
                   :, :, :, :, :, :, :, :, :
                   ].values,
            xi=xi,
        )
        at[l1.get_valid_mask(window)] = at_values
        atmosphere_components["aerosol_trans_total"][index, :, :] = at

        tot_sa_values = sp.interpn(
            points=lut_points,
            values=lut["spherical_albedo_total"][
                   :, :, :, :, :, :, :, :, :
                   ].values,
            xi=xi,
        )
        tot_sa[l1.get_valid_mask(window)] = tot_sa_values
        atmosphere_components["spherical_albedo_total"][index, :, :] = tot_sa


    return atmosphere_components

# def get_atmosphere(l1: ReveCube, tile, wavelength, image_sub):
#
#     x = image_sub.x.values
#     y = image_sub.y.values
#
#     atmosphere_components = xr.Dataset(
#         data_vars=dict(
#             gas_path_reflectance=(
#                 ["wavelength", "y", "x"],
#                 np.full((len(wavelength), len(y), len(x)), np.nan, float),
#             ),
#             gas_trans_total=(
#                 ["wavelength", "y", "x"],
#                 np.full((len(wavelength), len(y), len(x)), np.nan, float),
#             ),
#             rayleigh_trans_total=(
#                 ["wavelength", "y", "x"],
#                 np.full((len(wavelength), len(y), len(x)), np.nan, float),
#             ),
#             spherical_albedo_rayleigh=(
#                 ["wavelength", "y", "x"],
#                 np.full((len(wavelength), len(y), len(x)), np.nan, float),
#             ),
#             aer_path_reflectance=(
#                 ["wavelength", "y", "x"],
#                 np.full((len(wavelength), len(y), len(x)), np.nan, float),
#             ),
#             aerosol_trans_total=(
#                 ["wavelength", "y", "x"],
#                 np.full((len(wavelength), len(y), len(x)), np.nan, float),
#             ),
#             spherical_albedo_aerosol=(
#                 ["wavelength", "y", "x"],
#                 np.full((len(wavelength), len(y), len(x)), np.nan, float),
#             ),
#             spherical_albedo_total=(
#                 ["wavelength", "y", "x"],
#                 np.full((len(wavelength), len(y), len(x)), np.nan, float),
#             ),
#         ),
#         coords=dict(
#             wavelength=wavelength,
#             y=("y", y),
#             x=("x", x),
#         ),
#     )
#
#     # create placeholder for (x, y) gas_trans
#     gas_pr = np.full(l1.get_valid_mask(tile=tile).shape, np.nan, float)
#     gt = np.full(l1.get_valid_mask(tile=tile).shape, np.nan, float)
#     rt = np.full(l1.get_valid_mask(tile=tile).shape, np.nan, float)
#     gas_sa = np.full(l1.get_valid_mask(tile=tile).shape, np.nan, float)
#
#     aer_pr = np.full(l1.get_valid_mask(tile=tile).shape, np.nan, float)
#     at = np.full(l1.get_valid_mask(tile=tile).shape, np.nan, float)
#     aer_sa = np.full(l1.get_valid_mask(tile=tile).shape, np.nan, float)
#     tot_sa = np.full(l1.get_valid_mask(tile=tile).shape, np.nan, float)
#
#     # Do gas correction with 6S output "global_gas_trans_total"
#     # It integrates all gas transmittance in the downward and upward direction
#     # Dimensions of the LUT are {wavelength, sun_zenith, view_zenith, relative_azimuth,  sensor_altitude, water_vapor, ozone, target_pressure}
#     # Variables taken from the image {wavelength, sun_zenith, view_zenith, relative_azimuth, sensor_altitude}
#     # Variables taken from the GMAO MERRA2 model {water_vapor, ozone, target_pressure}
#
#     gas_lut = xr.open_dataset(
#         "/home/raphael/PycharmProjects/reverie/reverie/data/lut/gas_lut.nc"
#     )
#
#     aer_lut = xr.open_dataset(
#         "/home/raphael/PycharmProjects/reverie/reverie/data/lut/aerosol_lut.nc"
#     )
#
#     gas_points = (
#         gas_lut.sol_zen.values,
#         gas_lut.view_zen.values,
#         gas_lut.relative_azimuth.values,
#         gas_lut.water.values,
#         gas_lut.ozone.values,
#         gas_lut.target_pressure.values,
#         gas_lut.sensor_altitude.values,
#         gas_lut.wavelength.values,
#     )
#
#     aer_points = (
#         aer_lut.sol_zen.values,
#         aer_lut.view_zen.values,
#         aer_lut.relative_azimuth.values,
#         aer_lut.aot550.values,
#         aer_lut.target_pressure.values,
#         aer_lut.sensor_altitude.values,
#         aer_lut.wavelength.values,
#     )
#
#     n_pixels = len(
#         image_sub.variables["sun_zenith"]
#         .values[l1.get_valid_mask(tile=tile)]
#         .reshape(-1, 1)
#     )
#
#     for index, band in enumerate(wavelength):
#         gas_xi = np.hstack(
#             [
#                 image_sub.variables["sun_zenith"]
#                 .values[l1.get_valid_mask(tile=tile)]
#                 .reshape(-1, 1),
#                 image_sub.variables["view_zenith"]
#                 .values[l1.get_valid_mask(tile=tile)]
#                 .reshape(-1, 1),
#                 image_sub.variables["relative_azimuth"]
#                 .values[l1.get_valid_mask(tile=tile)]
#                 .reshape(-1, 1),
#                 np.repeat(
#                     image_sub.variables[
#                         "atmosphere_mass_content_of_water_vapor"
#                     ].values,
#                     n_pixels,
#                 ).reshape(-1, 1),
#                 np.repeat(
#                     image_sub.variables[
#                         "equivalent_thickness_at_stp_of_atmosphere_ozone_content"
#                     ].values,
#                     n_pixels,
#                 ).reshape(-1, 1),
#                 np.repeat(
#                     image_sub.variables["surface_air_pressure"].values, n_pixels
#                 ).reshape(-1, 1),
#                 np.repeat((-image_sub.z.values / 1000), n_pixels).reshape(-1, 1),
#                 np.repeat(band * 1e-3, n_pixels).reshape(-1, 1),
#             ]
#         )
#
#         gas_pr_values = sp.interpn(
#             points=gas_points,
#             # values=total_lut["atmospheric_reflectance_at_sensor"][
#             values=gas_lut["atmospheric_reflectance_at_sensor"][:, :, :, :, :, :, :, : ].values,
#             xi=gas_xi,
#         )
#
#         gas_pr[l1.get_valid_mask(tile=tile)] = gas_pr_values
#
#         atmosphere_components["gas_path_reflectance"][index, :, :] = gas_pr
#
#         gt_values = sp.interpn(
#             points=gas_points,
#             values=gas_lut["global_gas_trans_total"][
#                    :, :, :, :, :, :, :, :
#                    ].values,
#             xi=gas_xi,
#         )
#
#         gt[l1.get_valid_mask(tile=tile)] = gt_values
#
#         atmosphere_components["gas_trans_total"][index, :, :] = gt
#
#         rt_values = sp.interpn(
#             points=gas_points,
#             values=gas_lut["rayleigh_trans_total"][
#                    :, :, :, :, :, :, :, :
#                    ].values,
#             xi=gas_xi,
#         )
#
#         rt[l1.get_valid_mask(tile=tile)] = rt_values
#
#         atmosphere_components["rayleigh_trans_total"][index, :, :] = rt
#
#         # gas_sa_values = sp.interpn(
#         #     points=gas_points,
#         #     values=gas_lut["spherical_albedo_rayleigh"][
#         #            :, :, :, :, :, :, :, :
#         #            ].values,
#         #     xi=gas_xi,
#         # )
#         #
#         # gas_sa[l1.get_valid_mask(tile=tile)] = gas_sa_values
#         #
#         # atmosphere_components["spherical_albedo_rayleigh"][index, :, :] = gas_sa
#
#         aer_xi = np.hstack(
#             [
#                 image_sub.variables["sun_zenith"]
#                 .values[l1.get_valid_mask(tile=tile)]
#                 .reshape(-1, 1),
#                 image_sub.variables["view_zenith"]
#                 .values[l1.get_valid_mask(tile=tile)]
#                 .reshape(-1, 1),
#                 image_sub.variables["relative_azimuth"]
#                 .values[l1.get_valid_mask(tile=tile)]
#                 .reshape(-1, 1),
#                 np.repeat(
#                     image_sub.variables[
#                         "aerosol_optical_thickness_at_555_nm"
#                     ].values,
#                     n_pixels,
#                 ).reshape(-1, 1),
#                 np.repeat(
#                     image_sub.variables["surface_air_pressure"].values, n_pixels
#                 ).reshape(-1, 1),
#                 np.repeat((-image_sub.z.values / 1000), n_pixels).reshape(-1, 1),
#                 np.repeat(band * 1e-3, n_pixels).reshape(-1, 1),
#             ]
#         )
#
#         aer_pr_values = sp.interpn(
#             points=aer_points,
#             # values=total_lut["atmospheric_reflectance_at_sensor"][
#             values=aer_lut["atmospheric_reflectance_at_sensor"][:, :, :, :, :, :, :].values,
#             xi=aer_xi,
#         )
#
#         aer_pr[l1.get_valid_mask(tile=tile)] = aer_pr_values
#
#         atmosphere_components["aer_path_reflectance"][index, :, :] = aer_pr
#
#         at_values = sp.interpn(
#             points=aer_points,
#             values=aer_lut["aerosol_trans_total"][
#                    :, :, :, :, :, :, :
#                    ].values,
#             xi=aer_xi,
#         )
#
#         at[l1.get_valid_mask(tile=tile)] = at_values
#
#         atmosphere_components["aerosol_trans_total"][index, :, :] = at
#
#         aer_sa_values = sp.interpn(
#             points=aer_points,
#             values=aer_lut["spherical_albedo_aerosol"][
#                    :, :, :, :, :, :, :
#                    ].values,
#             xi=aer_xi,
#         )
#
#         aer_sa[l1.get_valid_mask(tile=tile)] = aer_sa_values
#
#         atmosphere_components["spherical_albedo_aerosol"][index, :, :] = aer_sa
#
#         tot_sa_values = sp.interpn(
#             points=aer_points,
#             values=aer_lut["spherical_albedo_total"][
#                    :, :, :, :, :, :, :
#                    ].values,
#             xi=aer_xi,
#         )
#
#         tot_sa[l1.get_valid_mask(tile=tile)] = tot_sa_values
#
#         atmosphere_components["spherical_albedo_total"][index, :, :] = tot_sa
#
#     return atmosphere_components
