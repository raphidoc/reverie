import math
import os

from scipy import stats
import xarray as xr
import pandas as pd
import numpy as np
import scipy.interpolate as sp
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from reverie import ReveCube
from reverie.correction.atmospheric.get_atmosphere import get_atmosphere
from reverie.correction.surface.get_rho_surface import get_rho_surface
from reverie.utils.helper import get_f0, mask_wavelength

import dsf
import lut

def run_ac(l1, gain):

    #TODO: Start by saving rho_t in it's own (temporary ?) file

    lut_aer = lut.load_aer()
    wavelength_lut = lut_aer["wavelength"].values

    l1.filter_bad_band()
    # filter image wavelength for LUT wavelength
    wavelength = mask_wavelength(l1.in_ds.wavelength.values, wavelength_lut)

    # Get extraterrestrial solar irradiance for WISE bands
    doy = l1.acq_time_z.timetuple().tm_yday
    f0 = get_f0(doy, wavelength)

    # Compute a dark spectrum of at sensor reflectance

    rho_dark = np.full_like(wavelength, np.nan)
    sol_zen = np.full_like(wavelength, np.nan)
    view_zen = np.full_like(wavelength, np.nan)
    relative_azimuth = np.full_like(wavelength, np.nan)

    # for i, wl in tqdm(enumerate(wavelength), ):
    #
    #     rho_t = (math.pi * l1.in_ds.sel(wavelength = wl)["radiance_at_sensor"]) / (
    #         f0[i] * np.cos(np.deg2rad(l1.in_ds["sun_zenith"]))
    #     )
    #
    #     rhot_t_sort = np.sort(rho_t.values, axis=None)
    #
    #     from scipy import stats
    #     n_pixels = 1000
    #
    #     slope, intercept, r, p, std_err = stats.linregress(list(range(n_pixels + 1)), rhot_t_sort[0:n_pixels+1])
    #
    #     rho_dark[i] = intercept

        # ### DEV
        # import matplotlib.pyplot as plt
        #
        # plt.imshow(rho_t, cmap='viridis')  # 'viridis' is a good default colormap
        # plt.colorbar(label='Slope Intensity')
        # plt.title('NDWI Matrix')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.show()
        #
        # plt.plot(rhot_t_sort[0:1000])
        # plt.xlabel("Index")
        # plt.ylabel("Value")
        # plt.title("Line Plot of a List")
        # plt.show()
        #
        # ### DEV

    args_list = [(i, wl, l1, f0) for i, wl in enumerate(wavelength)]
    results = process_map(dsf.compute_rho_dark, args_list, max_workers=os.cpu_count() -2)

    for i, sol_zen_val, view_zen_val, relative_azimuth_val, rho_dark_val in results:
        rho_dark[i] = rho_dark_val
        sol_zen[i] = sol_zen_val
        view_zen[i] = view_zen_val
        relative_azimuth[i] = relative_azimuth_val

    ### DEV
    import matplotlib.pyplot as plt

    plt.plot(wavelength, rho_dark)
    plt.plot(lut_aer.wavelength.values, lut_aer["atmospheric_reflectance_at_sensor"][0,0,2,0,0,0].values)
    plt.show()

    ### DEV

    # Interpolate rho_path to rho_dark and get the corresponding aod555 value
    # sol_zen = np.nanmean(l1.in_ds.sun_zenith.values)
    # view_zen = np.nanmean(l1.in_ds.view_zenith.values)
    # relative_azimuth = np.nanmean(l1.in_ds.relative_azimuth.values)
    target_pressure = l1.in_ds.surface_air_pressure.values
    sensor_altitude = l1.in_ds.z.values
    water = l1.in_ds.atmosphere_mass_content_of_water_vapor.values
    ozone = l1.in_ds.equivalent_thickness_at_stp_of_atmosphere_ozone_content

    aot555_dsf = dsf.aot555_dsf(
        wavelength[~np.isnan(sol_zen)],
        rho_dark[~np.isnan(sol_zen)],
        sol_zen[~np.isnan(sol_zen)],
        view_zen[~np.isnan(sol_zen)],
        relative_azimuth[~np.isnan(sol_zen)],
        target_pressure,
        sensor_altitude,
        water,
        ozone
    )

    # Apply atmospheric correction by wavelength on the entire image, write each band in the output dataset as l2r

    # Mask water with NDVI & NDWI

    # Apply sky and sun glint correction, write each band in the output dataset as l2r

    return 0

if __name__ == "__main__":
    # insitu data
    gain_path = "/D/Documents/phd/thesis/3_chapter/data/wise/viccal/gain_final.csv"
    # insitu_path = "/D/Documents/phd/thesis/3_chapter/data/wise/viccal/svc_rho.csv"
    gain = pd.read_csv(gain_path)

    image_dir = "/D/Data/WISE/"

    images = [
        # "ACI-10A/220705_ACI-10A-WI-1x1x1_v01-L1CG.nc",
        # "ACI-11A/220705_ACI-11A-WI-1x1x1_v01-L1CG.nc",
        # "ACI-12A/220705_ACI-12A-WI-1x1x1_v01-L1CG.nc",
        # "ACI-13A/220705_ACI-13A-WI-1x1x1_v01-L1CG.nc",
        "ACI-14A/220705_ACI-14A-WI-1x1x1_v01-L1CG.nc",
    ]

    for image in images:
        l1 = ReveCube.from_reve_nc(os.path.join(image_dir, image))

        run_ac(l1, gain)

