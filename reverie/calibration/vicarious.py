import os

import pandas
import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm
import math

from reverie import ReveCube
from reverie.correction.atmospheric.get_atmosphere import get_atmosphere
from reverie.correction.surface.get_rho_surface import get_rho_surface
from reverie.utils.helper import get_f0


import logging


def run_vicarious_cal(l1: ReveCube, in_situ: pandas.DataFrame, window_size, land: bool):
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

    # find the window on which we do the cal/val
    window_dict = l1.extract_pixel(
        insitu_path,
        max_time_diff=12,
        window_size=window_size,
        output_box = "/D/Documents/phd/thesis/3_chapter/data/wise/viccal/"
    )[1]

    if window_dict is None:
        raise Exception("No matchup found with in situ data")
    else:
        logging.info(f"Extracted window {list(window_dict.keys())}")

    # filter image for bad bands
    bad_band_list = image["radiance_at_sensor"].bad_band_list
    if isinstance(bad_band_list, str):
        bad_band_list = str.split(bad_band_list, ", ")

    bad_band_list = np.array(bad_band_list)
    good_band_indices = np.where(bad_band_list == "1")[0]
    good_bands_slice = slice(min(good_band_indices), max(good_band_indices) + 1)
    image = image.isel(wavelength=good_bands_slice)

    # filter common in situ, lut and image wavelength range
    min_wavelength_in_situ = in_situ["wavelength"].min()
    max_wavelength_in_situ = in_situ["wavelength"].max()

    lut = xr.open_dataset(
        "/home/raphael/PycharmProjects/reverie/reverie/data/lut/lut_combined.nc"
    )

    min_wavelength_lut = lut.wavelength.values.min() *1e3
    max_wavelength_lut = lut.wavelength.values.max() *1e3

    lut.close()

    min_wavelength = min(min_wavelength_in_situ, min_wavelength_lut)
    max_wavelength = min(max_wavelength_in_situ, max_wavelength_lut)

    # Create a boolean mask for the wavelengths within the range
    mask_image = (image.wavelength >= min_wavelength) & (
        image.wavelength <= max_wavelength
    )

    # Apply the mask to the image
    image = image.isel(wavelength=mask_image)
    wavelength = image.wavelength.values

    mask_in_situ = (in_situ["wavelength"] >= min_wavelength) &  (in_situ["wavelength"] <= max_wavelength)
    in_situ = in_situ[mask_in_situ]

    # Get extraterrestrial solar irradiance for WISE bands
    doy = l1.acq_time_z.timetuple().tm_yday
    f0 = get_f0(doy, wavelength)

    # test = pandas.DataFrame({"wavelength": wavelength,"f0": f0})
    # test.to_csv("/D/Documents/phd/thesis/3_chapter/data/wise/6S/f0_codington.csv")

    # Add x and y dimensions to f0
    f0_3d = f0[:, np.newaxis, np.newaxis]

    viccal_df = pd.DataFrame()

    for uuid, window in tqdm(window_dict.items()):
        # tile = Tile.FromWindow(window)

        # TODO: for some reason PME9 valid max return false
        #  even so the image_sub has data and is properly created by tile
        #  Reversed the index in get_valid_mask, check Yan code Tile.FromWindow
        y_start = int(window.row_off)
        y_stop = y_start + int(window.height)
        x_start = int(window.col_off)
        x_stop = x_start + int(window.width)

        image_sub = image.isel(y=slice(y_start, y_stop), x=slice(x_start, x_stop))
        # image_sub = image.isel(x=slice(tile[0], tile[1]), y=slice(tile[2], tile[3]))

        # # filter the image for the in situ wavelength range
        # min_wavelength_in_situ = in_situ["wavelength"].min()
        # max_wavelength_in_situ = in_situ["wavelength"].max()
        #
        # # Filter the image wavelengths based on these values
        # # Create a boolean mask for the wavelengths within the range
        # mask = (image_sub.wavelength >= min_wavelength_in_situ) & (
        #     image_sub.wavelength <= max_wavelength_in_situ
        # )
        #
        # # Apply the mask to the image
        # image_sub = image_sub.isel(wavelength=mask)

        # image["radiance_at_sensor"].mean(["x", "y"]).plot()
        # plt.show()

        if l1.valid_mask is None:
            l1.get_valid_mask(window=window)

        # Get dimension of image and create placeholder dataset for F0, atmosphere_components

        x = image_sub.x.values
        y = image_sub.y.values

        # Repeat f0 along the x and y dimension
        f0_image = np.repeat(f0_3d, len(y), axis=1)
        f0_image = np.repeat(f0_image, len(x), axis=2)

        f0_image = xr.Dataset(
            data_vars=dict(
                F0=(
                    ["wavelength", "y", "x"],
                    f0_image,
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

        # f0_image["F0"].mean(["x", "y"]).plot()
        # plt.show()
        #
        # test = image["radiance_at_sensor"] / f0_image["F0"]
        #
        # test.mean(["x", "y"]).plot()
        # plt.show()

        atmosphere_components = get_atmosphere(l1, window, wavelength, image_sub)

        trans_total = (
            atmosphere_components["gas_trans_total"]
            * atmosphere_components["rayleigh_trans_total"]
            * atmosphere_components["aerosol_trans_total"]
        )

        spherical_albedo = atmosphere_components["spherical_albedo_total"]

        # rho_path = atmosphere_components["atmospheric_reflectance_at_sensor"]
        rho_path = atmosphere_components["atmospheric_path_reflectance"]

        rho_w = (in_situ[in_situ["uuid"] == uuid]["rho"]).values
        in_situ_wl = in_situ[in_situ["uuid"] == uuid]["wavelength"].values

        rho_w = np.interp(x=wavelength, xp=in_situ_wl, fp=rho_w)

        # Add two new dimensions to temp
        rho_w_3d = rho_w[:, np.newaxis, np.newaxis]

        # Repeat temp along the new dimensions
        rho_w_3d = np.repeat(rho_w_3d, len(y), axis=1)
        rho_w_3d = np.repeat(rho_w_3d, len(x), axis=2)

        rho_w = xr.Dataset(
            data_vars=dict(
                rho_w=(
                    ["wavelength", "y", "x"],
                    rho_w_3d,
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
        # rho_s.mean(["x", "y"]).plot()
        # plt.show()

        if land:
            rho_sky = xr.Dataset(
                data_vars=dict(
                    rho_sky=(
                        ["wavelength", "y", "x"],
                        np.full((len(wavelength), len(y), len(x)), 0),
                    ),
                    rho_sun=(
                        ["wavelength", "y", "x"],
                        np.full((len(wavelength), len(y), len(x)), 0),
                    ),
                    rho=(
                        ["wavelength", "y", "x"],
                        np.full((len(wavelength), len(y), len(x)), 0),
                    ),
                ),
                coords=dict(
                    wavelength=wavelength,
                    y=("y", y),
                    x=("x", x),
                ),
            )
            rho_s = rho_w["rho_w"]
        else:
            rho_sky = get_rho_surface(image_sub, wavelength)
            rho_s = rho_w["rho_w"] + rho_sky["rho_surface_sky"]

        rho_t_target = rho_path + (rho_s * trans_total) / (1 - rho_s * spherical_albedo)

        rho_t = (math.pi * image_sub["radiance_at_sensor"]) / (
            f0_image["F0"] * np.cos(np.deg2rad(image_sub["sun_zenith"]))
        )

        gain = rho_t_target / rho_t

        # import matplotlib.pyplot as plt
        # # plt.rcParams["text.usetex"] = True
        #
        # rho_t_target.mean(["x", "y"]).plot(label="rho_t_target")
        #
        # # Calculate mean and standard deviation
        # rho_t_mean = rho_t.mean(["x", "y"])
        # rho_t_std = rho_t.std(["x", "y"])
        #
        # # Plot rho_t mean
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
        #
        # # Add LaTeX labels
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

        # gain_df = gain.to_dataframe()

        # Save the results to data frame
        temp = xr.merge(
            (image_sub, f0_image, atmosphere_components, rho_w, rho_sky, gain),
        )

        temp = temp.to_dataframe()

        temp["uuid"] = uuid
        temp["land"] = land

        viccal_df = pd.concat([viccal_df, temp], axis=0)

    return viccal_df


if __name__ == "__main__":
    # insitu data
    insitu_path = "/D/Documents/phd/thesis/3_chapter/data/wise/viccal/sas_rho.csv"
    # insitu_path = "/D/Documents/phd/thesis/3_chapter/data/wise/viccal/svc_rho.csv"
    in_situ = pd.read_csv(insitu_path)

    image_dir = "/D/Data/WISE/"

    images = [
        "ACI-10A/220705_ACI-10A-WI-1x1x1_v01-L1CG.nc",
        # "ACI-12A/220705_ACI-12A-WI-1x1x1_v01-L1CG.nc",
        # "ACI-13A/220705_ACI-13A-WI-1x1x1_v01-L1CG.nc",
        # "ACI-14A/220705_ACI-14A-WI-1x1x1_v01-L1CG.nc",
    ]

    for image in images:
        l1 = ReveCube.from_reve_nc(os.path.join(image_dir, image))

        viccal_df = run_vicarious_cal(l1, in_situ, window_size = 7, land = False)

        viccal_df.to_csv(
            os.path.join(
                "/D/Documents/phd/thesis/3_chapter/data/wise/viccal/",
                f"{l1.image_name}_gain.csv",
            )
        )
