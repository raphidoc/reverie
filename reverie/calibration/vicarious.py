import os

import pandas
import xarray as xr
import pandas as pd
import numpy as np
import scipy.interpolate as sp
from tqdm import tqdm
import math
import netCDF4 as nc

from reverie import ReveCube
from reverie.image.tile import Tile
from reverie.utils.helper import fwhm_2_rsr
from reverie.sky_glint import z17

import logging


def run_vicarious_cal(l1: ReveCube, in_situ: pandas.DataFrame, window_size):
    """
    Function to compute vicarious calibration gain from a ReveCube L1 image and in situ Rrs.

    Parameters
    ----------
    image
    in_situ
    window_size

    Returns
    -------

    Description
    -----------
    The gains are computed on apparent reflectance (rho_t^\star = Lt / F0).
    Path reflectance from 6S used in the calculation is also apparent reflectance.
    Therfore the underestimation of reflectance due to the overestimation of irradiance at the plane level cancels out.

    """

    image = l1.in_ds
    # filter image for bad bands
    bad_band_list = image["radiance_at_sensor"].bad_band_list
    if isinstance(bad_band_list, str):
        bad_band_list = str.split(bad_band_list, ", ")

    bad_band_list = np.array(bad_band_list)
    good_band_indices = np.where(bad_band_list == "1")[0]
    good_bands_slice = slice(min(good_band_indices), max(good_band_indices) + 1)
    image = image.isel(wavelength=good_bands_slice)

    # filter the image for the in situ wavelength range
    min_wavelength_in_situ = in_situ["wavelength"].min()
    max_wavelength_in_situ = in_situ["wavelength"].max()

    # Filter the image wavelengths based on these values
    # Create a boolean mask for the wavelengths within the range
    mask = (image.wavelength >= min_wavelength_in_situ) & (
        image.wavelength <= max_wavelength_in_situ
    )

    # Apply the mask to the image
    image = image.isel(wavelength=mask)
    wavelength = image.wavelength.values

    # Load F0 from coddington
    F0 = xr.open_dataset(
        "/home/raphael/PycharmProjects/reverie/reverie/data/solar_irradiance/hybrid_reference_spectrum_c2022-11-30_with_unc.nc"
    )

    # F0["SSI"][(F0["Vacuum Wavelength"] > 350) & (F0["Vacuum Wavelength"] < 1000)].plot()

    logging.info("Calculating equvilent F0")

    _waves = F0["Vacuum Wavelength"].values
    _f0 = F0["SSI"].values
    unit = F0["SSI"].units

    # calculate extraterrestrial solar irradiance on a specific day considering the sun-earth distance
    # (https://en.wikipedia.org/wiki/Sunlight#:~:text=If%20the%20extraterrestrial%20solar%20radiation,hitting%20the%20ground%20is%20around)
    doy = l1.acq_time_z.timetuple().tm_yday
    distance_factor = 1 + 0.033412 * np.cos(2 * np.pi * (doy - 3) / 365.0)
    _f0_cal = _f0 * distance_factor

    resolution, wrange = 0.01, (_waves[0], _waves[-1] + 0.0001)

    _f0_c_fine = sp.interp1d(_waves, _f0_cal)(
        np.arange(wrange[0], wrange[1], resolution)
    )

    # From https://stackoverflow.com/questions/3018758/determine-precision-and-scale-of-particular-number-in-python
    # def precision_and_scale(x):
    #     max_digits = 14
    #     int_part = int(abs(x))
    #     magnitude = 1 if int_part == 0 else int(math.log10(int_part)) + 1
    #     if magnitude >= max_digits:
    #         return (magnitude, 0)
    #     frac_part = abs(x) - int_part
    #     multiplier = 10 ** (max_digits - magnitude)
    #     frac_digits = multiplier + int(multiplier * frac_part + 0.5)
    #     while frac_digits % 10 == 0:
    #         frac_digits /= 10
    #     scale = int(math.log10(frac_digits))
    #     return (magnitude + scale, scale)
    #
    # precision_and_scale(F0["Vacuum Wavelength"][155])

    # Resolution is tricky to determine programmatically, hard code it
    resolution = 0.001

    rsr, wave = fwhm_2_rsr(
        np.full(len(wavelength), 5.05),
        wavelength,
        wrange=wrange,
        resolution=resolution,
    )
    f0_wise = np.zeros_like(wavelength, dtype=float)
    for i, item in enumerate(rsr):
        f0_wise[i] = np.sum(_f0_cal * item) / np.sum(item)

    # Simple interpolation to central wavelength doesn't work
    # temp = np.interp(
    #     x=wavelength, xp=F0["Vacuum Wavelength"].values, fp=F0["SSI"].values
    # )

    # Convert F0 from W m-2 nm-1 to uW cm-2 nm-1
    temp = f0_wise * 1e2

    # Add two new dimensions to temp (at the end)
    F0_3d = temp[:, np.newaxis, np.newaxis]

    # find the window on which we do the cal/val
    window_dict = l1.extract_pixel(
        insitu_path,
        max_time_diff=12,
        window_size=7,
    )[1]

    logging.info(f"Extracted window {window_dict.keys}")

    viccal_df = pd.DataFrame()

    for uuid, window in tqdm(window_dict.items()):
        tile = Tile.FromWindow(window)

        # TODO: for some reason PME9 valid max return false
        #  even so the image_sub has data and is properly created by tile
        #  Reversed the index in get_valid_mask, check Yan code Tile.FromWindow
        image_sub = image.isel(x=slice(tile[0], tile[1]), y=slice(tile[2], tile[3]))

        # filter the image for the in situ wavelength range
        min_wavelength_in_situ = in_situ["wavelength"].min()
        max_wavelength_in_situ = in_situ["wavelength"].max()

        # Filter the image wavelengths based on these values
        # Create a boolean mask for the wavelengths within the range
        mask = (image_sub.wavelength >= min_wavelength_in_situ) & (
            image_sub.wavelength <= max_wavelength_in_situ
        )

        # Apply the mask to the image
        image_sub = image_sub.isel(wavelength=mask)

        # image["radiance_at_sensor"].mean(["x", "y"]).plot()
        # plt.show()

        # Do gas correction with 6S output "global_gas_trans_total"
        # It integrates all gas transmittance in the downward and upward direction
        # Dimensions of the LUT are {wavelength, sun_zenith, view_zenith, relative_azimuth,  sensor_altitude, water_vapor, ozone, target_pressure}
        # Variables taken from the image {wavelength, sun_zenith, view_zenith, relative_azimuth, sensor_altitude}
        # Variables taken from the GMAO MERRA2 model {water_vapor, ozone, target_pressure}

        total_lut = xr.open_dataset(
            "/home/raphael/PycharmProjects/reverie/reverie/data/lut/output/test_total.nc"
        )

        if l1.valid_mask is None:
            l1.get_valid_mask(tile=tile)

        n_pixels = len(
            image_sub.variables["sun_zenith"]
            .values[l1.get_valid_mask(tile=tile)]
            .reshape(-1, 1)
        )

        # Get dimension of image and create placeholder dataset for F0, atmosphere_components

        x = image_sub.x.values
        y = image_sub.y.values

        # Repeat temp along the new dimensions
        temp_3d = np.repeat(F0_3d, len(y), axis=1)
        temp_3d = np.repeat(temp_3d, len(x), axis=2)

        F0_image = xr.Dataset(
            data_vars=dict(
                F0=(
                    ["wavelength", "y", "x"],
                    temp_3d,
                    {
                        "long_name": "Solar Spectral Irradiance Reference Spectrum (uW cm-2 nm-1)",
                        "standard_name": "solar_irradiance_per_unit_wavelength",
                        "units": "uW cm-2 nm-1",
                    },
                ),
            ),
            coords=dict(
                wavelength=wavelength,
                y=("y", y),
                x=("x", x),
            ),
        )

        # F0_image["F0"].mean(["x", "y"]).plot()
        # plt.show()
        #
        # test = image["radiance_at_sensor"] / F0_image["F0"]
        #
        # test.mean(["x", "y"]).plot()
        # plt.show()

        atmosphere_components = xr.Dataset(
            data_vars=dict(
                path_reflectance=(
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
                aerosol_trans_total=(
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

        # create placeholder for (x, y) gas_trans

        pr = np.full(l1.get_valid_mask(tile=tile).shape, np.nan, float)
        gt = np.full(l1.get_valid_mask(tile=tile).shape, np.nan, float)
        rt = np.full(l1.get_valid_mask(tile=tile).shape, np.nan, float)
        at = np.full(l1.get_valid_mask(tile=tile).shape, np.nan, float)
        sa = np.full(l1.get_valid_mask(tile=tile).shape, np.nan, float)

        points = (
            total_lut.sol_zen.values,
            total_lut.view_zen.values,
            total_lut.relative_azimuth.values,
            total_lut.water.values,
            total_lut.ozone.values,
            total_lut.aot550.values,
            total_lut.target_pressure.values,
            total_lut.sensor_altitude.values,
            total_lut.wavelength.values,
        )

        for index, band in enumerate(wavelength):
            xi = np.hstack(
                [
                    image_sub.variables["sun_zenith"]
                    .values[l1.get_valid_mask(tile=tile)]
                    .reshape(-1, 1),
                    image_sub.variables["view_zenith"]
                    .values[l1.get_valid_mask(tile=tile)]
                    .reshape(-1, 1),
                    image_sub.variables["relative_azimuth"]
                    .values[l1.get_valid_mask(tile=tile)]
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

            # Interpolate the values of global gas total transmittance
            pr_values = sp.interpn(
                points=points,
                # values=total_lut["atmospheric_reflectance_at_sensor"][
                values=total_lut["atmospheric_reflectance_at_sensor"][:, :, :, :, :, :, :, :, :].values,
                xi=xi,
            )

            pr[l1.get_valid_mask(tile=tile)] = pr_values

            atmosphere_components["path_reflectance"][index, :, :] = pr

            gt_values = sp.interpn(
                points=points,
                values=total_lut["global_gas_trans_total"][
                    :, :, :, :, :, :, :, :, :
                ].values,
                xi=xi,
            )

            gt[l1.get_valid_mask(tile=tile)] = gt_values

            atmosphere_components["gas_trans_total"][index, :, :] = gt

            rt_values = sp.interpn(
                points=points,
                values=total_lut["rayleigh_trans_total"][
                    :, :, :, :, :, :, :, :, :
                ].values,
                xi=xi,
            )

            rt[l1.get_valid_mask(tile=tile)] = rt_values

            atmosphere_components["rayleigh_trans_total"][index, :, :] = rt

            at_values = sp.interpn(
                points=points,
                values=total_lut["aerosol_trans_total"][
                    :, :, :, :, :, :, :, :, :
                ].values,
                xi=xi,
            )

            at[l1.get_valid_mask(tile=tile)] = rt_values

            atmosphere_components["aerosol_trans_total"][index, :, :] = at

            sa_values = sp.interpn(
                points=points,
                values=total_lut["spherical_albedo_total"][
                    :, :, :, :, :, :, :, :, :
                ].values,
                xi=xi,
            )

            sa[l1.get_valid_mask(tile=tile)] = sa_values

            atmosphere_components["spherical_albedo_total"][index, :, :] = sa

######### Rho sky z19

        rho_z17 = z17.get_sky_sun_rho(
            aot_550=float(image_sub.variables["aerosol_optical_thickness_at_555_nm"].mean().values),
            sun_zen=float(image_sub.variables["sun_zenith"].mean().values),
            view_ang=[float(image_sub.variables["view_zenith"].mean().values), float(image_sub.variables["relative_azimuth"].mean().values)],
            water_salinity=30,
            water_temperature=17,
            wavelength=wavelength,
            wind_speed=float(image_sub.variables["wind_speed"].mean().values)
        )

        sky = rho_z17["sky"]
        # Add two new dimensions to temp
        sky_3d = sky[:, np.newaxis, np.newaxis]

        # Repeat temp along the new dimensions
        sky_3d = np.repeat(sky_3d, len(y), axis=1)
        sky_3d = np.repeat(sky_3d, len(x), axis=2)

        sun = rho_z17["sun"]
        # Add two new dimensions to temp
        sun_3d = sun[:, np.newaxis, np.newaxis]

        # Repeat temp along the new dimensions
        sun_3d = np.repeat(sun_3d, len(y), axis=1)
        sun_3d = np.repeat(sun_3d, len(x), axis=2)

        rho = rho_z17["rho"]
        # Add two new dimensions to temp
        rho_3d = rho[:, np.newaxis, np.newaxis]

        # Repeat temp along the new dimensions
        rho_3d = np.repeat(rho_3d, len(y), axis=1)
        rho_3d = np.repeat(rho_3d, len(x), axis=2)

        rho_sky = xr.Dataset(
            data_vars=dict(
                rho_sky=(
                    ["wavelength", "y", "x"],
                    sky_3d,
                ),
                rho_sun=(
                    ["wavelength", "y", "x"],
                    sun_3d,
                ),
                rho=(
                    ["wavelength", "y", "x"],
                    rho_3d,
                ),
            ),
            coords=dict(
                wavelength=wavelength,
                y=("y", y),
                x=("x", x),
            ),
        )

        ######### Rho sky OSOAA

        # rho_sky = xr.Dataset(
        #     data_vars=dict(
        #         rho_sky=(
        #             ["wavelength", "y", "x"],
        #             np.full((len(wavelength), len(y), len(x)), np.nan, float),
        #         ),
        #     ),
        #     coords=dict(
        #         wavelength=wavelength,
        #         y=("y", y),
        #         x=("x", x),
        #     ),
        # )
        #
        # # rho_sky_nc = xr.open_dataset(
        # #     filename_or_obj="/D/Documents/phd/thesis/2_chapter/data/ACOLITE-RSKY-202102-82W-MOD2.nc",
        # #     engine='netcdf4'
        # # )
        #
        # rho_sky_nc = nc.Dataset(
        #     "/D/Documents/phd/thesis/2_chapter/data/ACOLITE-RSKY-202102-82W-MOD2.nc"
        # )
        #
        # # Compute sea surface reflection from the ACOLITE Lut (OSOAA model)
        #
        # sr = np.full(l1.get_valid_mask(tile=tile).shape, np.nan, float)
        #
        # for index, band in enumerate(wavelength):
        #     xi = np.hstack(
        #         [
        #             np.repeat(band * 1e-3, n_pixels).reshape(-1, 1),
        #             image_sub.variables["relative_azimuth"]
        #             .values[l1.get_valid_mask(tile=tile)]
        #             .reshape(-1, 1),
        #             image_sub.variables["view_zenith"]
        #             .values[l1.get_valid_mask(tile=tile)]
        #             .reshape(-1, 1),
        #             image_sub.variables["sun_zenith"]
        #             .values[l1.get_valid_mask(tile=tile)]
        #             .reshape(-1, 1),
        #             np.repeat(
        #                 image_sub.variables["wind_speed"].values,
        #                 n_pixels,
        #             ).reshape(-1, 1),
        #             np.repeat(
        #                 image_sub.variables[
        #                     "aerosol_optical_thickness_at_555_nm"
        #                 ].values,
        #                 n_pixels,
        #             ).reshape(-1, 1),
        #         ]
        #     )
        #
        #     sea_rho = sp.interpn(
        #         points=(
        #             rho_sky_nc.wave[()],
        #             rho_sky_nc.azi[()],
        #             rho_sky_nc.thv[()],
        #             rho_sky_nc.ths[()],
        #             rho_sky_nc.wind[()],
        #             rho_sky_nc.tau[()],
        #             # rho_sky_nc.attrs["wave"],
        #             # rho_sky_nc.attrs["azi"],
        #             # rho_sky_nc.attrs["thv"],
        #             # rho_sky_nc.attrs["ths"],
        #             # rho_sky_nc.attrs["wind"],
        #             # rho_sky_nc.attrs["tau"]
        #         ),
        #         values=rho_sky_nc.variables["sky_glint"][:, :, :, :, :, :],
        #         xi=xi,
        #         method="linear"
        #     )
        #
        #     sr[l1.get_valid_mask(tile=tile)] = sea_rho
        #
        #     rho_sky["rho_sky"][index, :, :] = sr

        total_trans = (
            atmosphere_components["gas_trans_total"]
            * atmosphere_components["rayleigh_trans_total"]
            * atmosphere_components["aerosol_trans_total"]
        )

        spherical_albedo = atmosphere_components["spherical_albedo_total"]

        # rho_path = atmosphere_components["atmospheric_reflectance_at_sensor"]
        rho_path = atmosphere_components["path_reflectance"]

        rho_w = (in_situ[in_situ["uuid"] == uuid]["rrs_mean"] * math.pi).values
        in_situ_wl = in_situ[in_situ["uuid"] == uuid]["wavelength"].values

        temp = np.interp(x=wavelength, xp=in_situ_wl, fp=rho_w)

        # Add two new dimensions to temp
        temp_3d = temp[:, np.newaxis, np.newaxis]

        # Repeat temp along the new dimensions
        temp_3d = np.repeat(temp_3d, len(y), axis=1)
        temp_3d = np.repeat(temp_3d, len(x), axis=2)

        rho_w = xr.Dataset(
            data_vars=dict(
                rho_w=(
                    ["wavelength", "y", "x"],
                    temp_3d,
                ),
            ),
            coords=dict(
                wavelength=wavelength,
                y=("y", y),
                x=("x", x),
            ),
        )

        # rho_w["rho_w"].mean(["x", "y"]).plot()
        # plt.show()

        rho_s = rho_w["rho_w"] + rho_sky["rho_sky"]
        # rho_s.mean(["x", "y"]).plot()
        # plt.show()

        rho_t_target = rho_path + (rho_s * total_trans) / (1 - rho_s * spherical_albedo)

        rho_t = (math.pi * image_sub["radiance_at_sensor"]) / (
            F0_image["F0"] * np.cos(np.deg2rad(image_sub["sun_zenith"]))
        )

        gain = rho_t_target / rho_t

        # plt.rcParams["text.usetex"] = True

        # rho_t_target.mean(["x", "y"]).plot(label="rho_t_target")

        # Calculate mean and standard deviation
        # rho_t_mean = rho_t.mean(["x", "y"])
        # rho_t_std = rho_t.std(["x", "y"])

        # Plot rho_t mean
        # rho_t_mean.plot(label="rho_t")
        #
        # # Add standard deviation shading
        # plt.fill_between(
        #     rho_t_mean.wavelength,
        #     rho_t_mean - rho_t_std,
        #     rho_t_mean + rho_t_std,
        #     color="gray",
        #     alpha=0.2,
        # )
        #
        # plt.xlabel("Wavelength [nm]")
        # plt.ylabel("Reflectance")
        # plt.legend()
        # plt.show()

        # Add LaTeX labels
        # plt.xlabel(r"$\text{Wavelength}\ [\text{nm}]$")
        # plt.ylabel(r"$\text{Reflectance}$")
        # plt.xlabel(r"Wavelength [nm]")
        # plt.ylabel(r"Reflectance")
        #
        # plt.legend()
        # plt.show()
        #
        # gain_mean.plot()
        #
        # plt.fill_between(
        #     gain_mean.wavelength,
        #     gain_mean - gain_std,
        #     gain_mean + gain_std,
        #     color="gray",
        #     alpha=0.2,
        # )
        #
        # plt.xlabel(r"Wavelength [nm]")
        # plt.ylabel(r"Gain = rho_t_target / rho_t")
        #
        # plt.show()

        # And now re atmospheric correction just to check that we indeed retrive the same values
        # rho_t_cal = rho_t * gain
        # # rho_t_cal.mean(["x", "y"]).plot()
        # # plt.show()
        #
        #
        # rho_s_cal = (rho_t_cal - rho_path) / (
        #     atmosphere_components + spherical_albedo * (rho_t_cal - rho_path)
        # )
        # # rho_s_cal.mean(["x", "y"]).plot()
        # # plt.show()
        #
        # rho_w_cal = rho_s_cal - rho_sky["rho_sky"]
        #
        # rho_w_cal.mean(["x", "y"]).plot()
        # plt.show()

        gain.name = "gain"

        gain_df = gain.to_dataframe()

        # Save the results to data frame
        temp = xr.merge(
            (image_sub, F0_image, atmosphere_components, rho_w, rho_sky, gain),
        )

        temp = temp.to_dataframe()

        temp["uuid"] = uuid

        viccal_df = pd.concat([viccal_df, temp], axis=0)

    return viccal_df


if __name__ == "__main__":
    # insitu data
    insitu_path = "/D/Documents/phd/thesis/2_chapter/data/wise/viccal/viccal_rrs.csv"
    in_situ = pd.read_csv(insitu_path)

    image_dir = "/D/Data/WISE/"

    images = [
        "ACI-10A/220705_ACI-10A-WI-1x1x1_v01-L1CG.nc",
        # "ACI-11A/220705_ACI-11A-WI-1x1x1_v01-L1CG.nc",
        "ACI-12A/220705_ACI-12A-WI-1x1x1_v01-L1CG.nc",
        "ACI-13A/220705_ACI-13A-WI-1x1x1_v01-L1CG.nc",
        # "MC-50A/190818_MC-50A-WI-2x1x1_v02-L1CG.nc",
        # "MC-37A/190818_MC-37A-WI-1x1x1_v02-L1CG.nc",
        # "MC-10A/190820_MC-10A-WI-1x1x1_v02-L1G.nc",
    ]

    for image in images:
        l1 = ReveCube.from_reve_nc(os.path.join(image_dir, image))

        viccal_df = run_vicarious_cal(l1, in_situ, 7)

        viccal_df.to_csv(
            os.path.join(
                "/D/Documents/phd/thesis/2_chapter/data/wise/viccal/",
                f"{l1.image_name}_gain.csv",
            )
        )
