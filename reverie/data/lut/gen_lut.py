"""
Script used to generate the lookup table for the atmospheric correction
The LUT from acolite is used as a starting point:

* 'wave' (wavelength [um], size = 82):
    [0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4 , 0.41, 0.42, 0.43, 0.44, 0.45,
    0.46, 0.47, 0.48, 0.49, 0.5 , 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57,
    0.58, 0.59, 0.6 , 0.61, 0.62, 0.63, 0.64,0.65, 0.66, 0.67, 0.68, 0.69,
    0.7 , 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8 , 0.81,
    0.82, 0.83,0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9 , 1., 1.1 , 1.2 , 1.3,
    1.4, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 2., 2.05, 2.1,
    2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5]

* 'azi' (relative azimuth [degree], size = 13):
    [0., 10., 20., 40., 60., 80., 90., 100., 120., 140., 160., 170., 180.]

* 'thv' (viewing zenith [degree], size = 13):
    [0., 1.5,  4., 8., 12., 16., 24., 32., 40., 48., 56., 64., 72.]

* 'ths' (sun zenith [degree], size = 16):
    [0.,  1.5,  4.,  8., 16., 24., 32., 40., 48., 56., 60., 64., 68., 72., 76., 80.]

* 'wnd' (wind speed [m s-1], size = 1):
    2.0

* 'tau' (aerosol optical thikness at 550, size = 16):
    [1.0e-03, 1.0e-02, 2.0e-02, 5.0e-02, 1.0e-01, 1.5e-01, 2.0e-01, 3.0e-01, 5.0e-01, 7.0e-01,1.0e+00, 1.3e+00, 1.6e+00, 2.0e+00, 3.0e+00, 5.0e+00]

* Pressure at target [mbar] (proxy of target altitude) (size = 4):
    As given in the aerosol filename [500., 750., 1013., 1100]


On which 19 parameters are indexed as: lut(par, wave, azi, thv, ths, wnd, tau)

* 'wl' (wavelength)
* 'utotr' (upwelling total Rayleigh transmittance)
* 'dtotr' (downwelling total Rayleigh transmittance)
* 'phar' Not used
* 'asray' (spherical albedo Rayleigh)
* 'tray' (transmittance Rayleigh)
* 'rorayl' (reflectance Rayleigh (6S composante I))
* 'utota' (upwelling total aerosol transmittance)
* 'dtota' (downwelling total aerosol transmittance)
* 'phaa' Not used
* 'asaer' (spherical albedo aerosol)
* 'taer' (transmittance aerosol ?)
* 'roaero' (reflectance aerosol (6S composante I))
* 'utott' (upwelling total transmittance)
* 'dtott' (downwelling total transmittance)
* 'astot' (spherical albedo total)
* 'romix' (path reflectance, Rayleigh + aerosol)
* 'roc' Not used
* 'rsurf' (surface reflectance, Fresnel ?)

The water vapor LUT is generated using the following parameters:
* 'wv':
    [0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4.]

"""

import os
import subprocess

import netCDF4
import numpy as np
from p_tqdm import p_uimap


def create_gas_output_nc(filename, coords, compression=None, complevel=None):
    # Create the output netcdf file
    nc = netCDF4.Dataset(
        filename,
        "w",
        format="NETCDF4",
    )
    # Create the dimensions

    nc.createDimension("wavelength", len(coords["wavelengths"]))
    nc.createDimension("sol_zen", 1)
    nc.createDimension("view_zen", 1)
    nc.createDimension("raa", 1)
    nc.createDimension("target_pressure", 1)
    nc.createDimension("sensor_altitude", 1)
    nc.createDimension("water", 1)
    nc.createDimension("ozone", 1)

    # Create the coordinates variables
    wavelength_nc = nc.createVariable("wavelength", "f4", ("wavelength",))
    wavelength_nc.standard_name = "radiation_wavelength"
    wavelength_nc.units = "nm"
    wavelength_nc[:] = coords["wavelengths"]

    sol_zen_nc = nc.createVariable("sol_zen", "f4", ("sol_zen",))
    sol_zen_nc.standard_name = "solar_zenith_angle"
    sol_zen_nc.units = "degree"
    sol_zen_nc[:] = coords["sol_zen"]

    view_zen_nc = nc.createVariable("view_zen", "f4", ("view_zen",))
    view_zen_nc.standard_name = "sensor_zenith_angle"
    view_zen_nc.units = "degree"
    view_zen_nc[:] = coords["view_zen"]

    raa_nc = nc.createVariable("raa", "f4", ("raa",))
    raa_nc.standard_name = "angle_of_rotation_from_solar_azimuth_to_platform_azimuth"
    raa_nc.long_name = "relative_azimuth_angle"
    raa_nc.units = "degree"
    raa_nc[:] = coords["raa"]

    pressure_nc = nc.createVariable("target_pressure", "f4", ("target_pressure",))
    pressure_nc.units = "mbar"
    pressure_nc[:] = coords["target_pressure"]

    altitude_nc = nc.createVariable("sensor_altitude", "f4", ("sensor_altitude",))
    altitude_nc.standard_name = "altitude"
    altitude_nc.units = "km"
    altitude_nc[:] = coords["sensor_altitude"]

    water_nc = nc.createVariable("water", "f4", ("water",))
    water_nc.standard_name = "atmosphere_mass_content_of_water_vapor"
    water_nc.units = "g cm-2"
    water_nc[:] = coords["water"]

    ozone_nc = nc.createVariable("ozone", "f4", ("ozone",))
    ozone_nc.standard_name = "equivalent_thickness_at_stp_of_atmosphere_ozone_content"
    ozone_nc.units = "atm-cm"
    ozone_nc[:] = coords["ozone"]

    # Create the data variables
    dimensions = (
        "wavelength",
        "sol_zen",
        "view_zen",
        "raa",
        "target_pressure",
        "sensor_altitude",
        "water",
        "ozone",
    )

    nc.createVariable(
        "tgv", "f4", dimensions, compression=compression, complevel=complevel
    )
    nc.createVariable(
        "tgs", "f4", dimensions, compression=compression, complevel=complevel
    )

    return nc


def run_gaseous_sim(params: list):
    """
    Run the 6S simulation for the gaseous LUT


    Parameters
    ----------
    params: list
        List of parameters to pass to the simulation with order:
        sol_zen, view_zen, raa, water, ozone, target_pressure, sensor_altitude, wavelength

    Returns
    -------

    """

    # t0 = time.perf_counter()

    kwargs = {}

    input_str = "\n".join(
        [
            # Geometrical conditions
            "0 # Geometrical conditions igeom",
            f"{params[0]} 0.0 {params[1]} {params[2]} 1 1",
            # Atmospheric conditions, gas
            "8 # Atmospheric conditions, gas (idatm)",
            f"{params[3]} {params[4]}",
            # Aerosol type and concentration
            "0 #iaer No aerosol",
            "-1 # Visibility (km)",
            # Target altitude as pressure in mb
            f"{params[5]} # target altitude (xps)",
            # Sensor altitude in km relative to target altitude
            f"{params[6]} # Sensor altitude (xpp)",
            # If aircraft simulation (> -110 < 0) enter water vapor, ozone content
            # Aircraft water vapor, ozone and aot550
            "-1.0 -1.0",
            "-1.0",
            # Spectrum selection for simulation
            "-1 # Monochromatic spectrum selection (iwave)",
            f"{params[7]}",
            # Ground reflectance
            "0 # Ground reflectance (inhomo)",
            "0",
            "0",
            "0.000",
            # Atmospheric correction
            "-1 # Atmospheric correction (irapp)",
            "",
        ]
    )

    exec_6sv2 = "/home/raphael/PycharmProjects/6sV2.1/sixsV2.1"
    process = subprocess.run(exec_6sv2, input=input_str, text=True, capture_output=True)

    # process.stderr
    # process.stdout

    # Return the results in a dictionary with parameters
    # return {
    #     "coords": {
    #         "wavelengths": wls,
    #         "sol_zen": solar_z,
    #         "view_zen": view_z,
    #         "raa": raa,
    #         "target_pressure": target_pressure,
    #         "sensor_altitude": sensor_altitude,
    #         "water": water,
    #         "ozone": ozone,
    #     },
    #     "tgs": ,
    #     "tgv": ,
    # }


def run_rayleigh_aerosol_sim(
    wavelengths,
    solar_z,
    view_z,
    raa,
    target_pressure,
    sensor_altitude,
    aot550,
):
    return


if __name__ == "__main__":
    # common LUT dimensions using ATCOR
    sol_zen_dim = np.arange(0, 70, 10).tolist()  # degree
    # View zenith of WISE doesnt go above 24 degrees
    view_zen_dim = np.arange(0, 40, 10).tolist()  # degree
    raa_dim = np.arange(0.0, 180.0, 30).tolist()  # degree
    target_pres_dim = [750.0, 1013.0, 1100]  # mbar
    # Sensor altitude at -1000 is used by 6S to indicate satellite altitude
    sensor_alt_dim = [
        -1,
        -3,
        -1000,
    ]  # km
    # These are the 20 node wavelengths used by 6S
    wavelength_dim = [
        0.350,
        0.400,
        0.412,
        0.443,
        0.470,
        0.488,
        0.515,
        0.550,
        0.590,
        0.633,
        0.670,
        0.694,
        0.760,
        0.860,
        1.240,
        1.536,
        1.650,
        1.950,
        2.250,
        3.750,
    ]

    # Gaseous LUT dimensions
    water_dim = np.arange(0.0, 4.0, 0.5).tolist()  # g cm-2
    ozone_dim = np.arange(0.0, 0.5, 0.05).tolist()  # atm-cm

    # Create an iterator for the parameters to pass to the gas simulation
    # Use of a dictionary would be more readable
    # order must be the same as in the input_str
    # sol_zen, view_zen, raa, water, ozone, target_pressure, sensor_altitude, wavelength
    param_iter = np.array(
        np.meshgrid(
            sol_zen_dim,
            view_zen_dim,
            raa_dim,
            water_dim,
            ozone_dim,
            target_pres_dim,
            sensor_alt_dim,
            wavelength_dim,
        )
    ).T.reshape(-1, 8, order="C")

    # add the wavelength dimension to the parameter iterator
    # Not working, create an array with memory size ~ 52.9 TiB
    # wavelength_array = np.repeat(
    #     np.array(wavelength_dim).reshape(-1, 1), param_iter.shape[0], axis=1
    # ).T
    # param_iter = np.insert(param_iter, 0, wavelength_array, axis=1)

    combination = (
        len(sol_zen_dim)
        * len(view_zen_dim)
        * len(raa_dim)
        * len(target_pres_dim)
        * len(sensor_alt_dim)
        * len(water_dim)
        * len(ozone_dim)
    )

    print(
        f"{combination} parameters combinations to simulate {combination * 15 / 3600:.2f}h"
    )

    lut_dir = "/home/raphael/PycharmProjects/reverie/reverie/data/lut"

    output_dir = os.path.join(lut_dir, "output")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Create a pool of workers
    # with Pool() as p:
    #     results = tqdm(p.imap_unordered(run_simulation, param_iter[0:1]))

    iterator = p_uimap(run_gaseous_sim, param_iter)

    # Write the results to separate files
    for i, result in enumerate(iterator):
        continue

    #     filename = f"gas_output_{i}.nc"
    #
    #     coords = result["coords"]
    #
    #     nc = create_gas_output_nc(os.path.join(output_dir, filename), coords)
    #
    #     # Write the results to the file
    #     nc.variables["tgs"][
    #         :,
    #         :,
    #         :,
    #         :,
    #         :,
    #         :,
    #         :,
    #         :,
    #     ] = result["tgs"]
    #
    #     nc.variables["tgv"][
    #         :,
    #         :,
    #         :,
    #         :,
    #         :,
    #         :,
    #         :,
    #         :,
    #     ] = result["tgv"]
    #
    #     nc.close()
    #
    # # Merge the files
    # files = [os.path.join(output_dir, file) for file in os.listdir(output_dir)]
    # combined = xr.open_mfdataset(files)
    # combined.to_netcdf(os.path.join(lut_dir, "combined.nc"))

# for i in tqdm(range(param_iter.shape[0])):
#     print(f"\r\nSimulating {i + 1} / {param_iter.shape[0]}")
#     # Get the parameters
#     (
#         sol_zen,
#         view_zen,
#         raa,
#         target_pressure,
#         sensor_altitude,
#         water,
#         ozone,
#     ) = param_iter[i]
#
#     # Run the simulation
#     wls, res = run_gaseous_sim(
#         wavelength_dim,
#         sol_zen,
#         view_zen,
#         raa,
#         target_pressure,
#         sensor_altitude,
#         water,
#         ozone,
#     )
#
#     # Check some results in original format
#     # with open("/home/raphael/PycharmProjects/reverie/reverie/data/lut/6S_output.txt", 'w') as f:
#     #     f.write(res_df['fulltext'][0])
#
#     # Write the results to the LUT
#     tgs_nc[
#         :,
#         sol_zen_dim.index(sol_zen),
#         view_zen_dim.index(view_zen),
#         raa_dim.index(raa),
#         target_pres_dim.index(target_pressure),
#         sensor_alt_dim.index(sensor_altitude),
#         water_dim.index(water),
#         ozone_dim.index(ozone),
#     ] = [
#         r.transmittance_global_gas.downward for r in res
#     ]  # res_df["global_gas_downward"]
#
#     tgv_nc[
#         :,
#         sol_zen_dim.index(sol_zen),
#         view_zen_dim.index(view_zen),
#         raa_dim.index(raa),
#         target_pres_dim.index(target_pressure),
#         sensor_alt_dim.index(sensor_altitude),
#         water_dim.index(water),
#         ozone_dim.index(ozone),
#     ] = [
#         r.transmittance_global_gas.upward for r in res
#     ]  # res_df["global_gas_upward"]

# Rayleigh aerosol LUT dimensions
# aot550_dim = [
#     0.001,
#     0.01,
#     0.02,
#     0.05,
#     0.1,
#     0.15,
#     0.2,
#     0.3,
#     0.5,
#     0.7,
#     1,
#     1.3,
#     1.6,
#     2,
#     3,
#     5,
# ]
