import math

from scipy import stats
import xarray as xr
import numpy as np
import scipy.interpolate as sp
from tqdm import tqdm

import lut

def compute_rho_dark(args):
    i, wl, l1, f0 = args
    rho_t = (math.pi * l1.in_ds.sel(wavelength=wl)["radiance_at_sensor"]) / (
            f0[i] * np.cos(np.deg2rad(l1.in_ds["sun_zenith"]))
    )
    # rho_t_sort = np.sort(rho_t.values, axis=None)
    # n_pixels = 100
    # slope, intercept, r, p, std_err = stats.linregress(
    #     list(range(n_pixels + 1)), rho_t_sort[0:n_pixels + 1]
    # )

    # min_idx = np.nanargmin(rho_t.values)

    percentile = 0.01  # Change to desired percentile
    rho_dark = np.nanpercentile(rho_t.values, percentile)
    i_dark = np.argwhere(rho_t.values == rho_dark)
    # if i_dark.size > 0:
    #     i_dark = tuple(i_dark[0])
    # if i_dark.size == 0 :
    #     i_dark = tuple(i_dark)
    # else:
    #     i_dark = None

    i_dark = tuple(i_dark[0]) if i_dark.size > 0 else None

    # Extract corresponding values
    sol_zen = l1.in_ds["sun_zenith"].values[i_dark] if i_dark is not None else np.nan
    view_zen = l1.in_ds["view_zenith"].values[i_dark] if i_dark is not None else np.nan
    relative_azimuth = l1.in_ds["relative_azimuth"].values[i_dark] if i_dark is not None else np.nan

    return i, sol_zen, view_zen, relative_azimuth, rho_dark

def aot555_dsf(wavelength, rho_dark, sol_zen, view_zen, relative_azimuth, target_pressure, sensor_altitude, water, ozone):

    lut_aer = lut.load_aer()

    lut_points = (
        lut_aer.sol_zen.values,
        lut_aer.view_zen.values,
        lut_aer.relative_azimuth.values,
        lut_aer.aot550.values,
        lut_aer.target_pressure.values,
        lut_aer.sensor_altitude.values,
        lut_aer.wavelength.values,
    )

    aot550_values = lut_aer.aot550.values
    n_aot = len(aot550_values)

    rho_path_corrected = np.full((len(wavelength), n_aot), np.nan)
    aot550_candidate = np.full_like(wavelength, np.nan)
    for i, wl in tqdm(enumerate(wavelength)):
        # TODO: apply gas transmittance "correction" to rho_path
        # Innacurate because doesn't account for true interaction between scattering and absorption
        # 6S rho_path(Rayleigh + aerosol + gas) !=
        # (rho_path(rayleigh+aerosol) * t_gas) / (1 - rho_path(Rayleigh+aerosol) * Spherical albedo)
        # Viccarious calibration "remove" this innacuracy



        rho_path_values = lut_aer["atmospheric_reflectance_at_sensor"][:, :, :, :, :, :, :].values
        s_values = lut_aer["spherical_albedo_total"][:, :, :, :, :, :, :].values


        xi = np.hstack([
            np.full((n_aot, 1), sol_zen[i]),
            np.full((n_aot, 1), view_zen[i]),
            np.full((n_aot, 1), relative_azimuth[i]),
            aot550_values.reshape(-1, 1),
            np.full((n_aot, 1), target_pressure),
            np.full((n_aot, 1), sensor_altitude),
            np.full((n_aot, 1), wl),
        ])

        rho_path = sp.interpn(
            points=lut_points,
            # values=lut_aer["atmospheric_reflectance_at_sensor"][:, :, :, :, :, :, :].values
            values=rho_path_values,
            xi=xi,
        )

        s = sp.interpn(
            points=lut_points,
            # values=lut_aer["atmospheric_reflectance_at_sensor"][:, :, :, :, :, :, :].values
            values=s_values,
            xi=xi,
        )

        t_gas = lut.get_t_gas(wl, sol_zen[i], view_zen[i], relative_azimuth[i], water, ozone, target_pressure,
                              sensor_altitude)

        rho_path_corrected[i] = (rho_path * t_gas) / (1 - rho_path * s)

        f_interp = sp.interp1d(rho_path_corrected[i], aot550_values, bounds_error=False, fill_value=np.nan)
        aot550_candidate[i] = f_interp(rho_dark[i])

    ### DEV

    import matplotlib.pyplot as plt

    aot_index = 1
    plt.plot(wavelength, rho_dark)
    plt.plot(wavelength, rho_path_corrected[:, aot_index])  # aot_index: integer index for desired AOT
    plt.show()

    plt.plot(aot550_candidate)
    plt.show()

    ### DEV
    aot550_retrieved = np.nanmedian(aot550_candidate)

    return aot550_retrieved