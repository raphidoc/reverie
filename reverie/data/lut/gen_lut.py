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

import concurrent.futures
import itertools
import os
import subprocess
import time

import netCDF4
import numpy as np
from p_tqdm import p_uimap
import json


def format_estimated_time(estimated_time):
    if estimated_time >= 60.0 * 60 * 24 * 365 * 100:
        return str(estimated_time / (60.0 * 60 * 24 * 365 * 100)) + " centuries"
    elif estimated_time >= 60.0 * 60 * 24 * 365:
        return str(estimated_time / (60.0 * 60 * 24 * 365)) + " years"
    elif estimated_time >= 60 * 60 * 24 * 30:
        return str(estimated_time / (60 * 60 * 24 * 30)) + " months"
    elif estimated_time >= 60 * 60 * 24 * 7:
        return str(estimated_time / (60 * 60 * 24 * 7)) + " weeks"
    elif estimated_time >= 60 * 60 * 24:
        return str(estimated_time / (60 * 60 * 24)) + " days"
    elif estimated_time >= 60 * 60:
        return str(estimated_time / (60 * 60)) + " hours"
    elif estimated_time >= 60:
        return str(estimated_time / 60) + " minutes"
    else:
        return str(estimated_time) + " seconds"


def create_gas_output_nc(filename, coords, compression=None, complevel=None):
    # Create the output netcdf file
    nc = netCDF4.Dataset(
        filename,
        "w",
        format="NETCDF4",
    )
    # Create the dimensions

    nc.createDimension("sol_zen", len(coords["sol_zen"]))
    nc.createDimension("view_zen", len(coords["view_zen"]))
    nc.createDimension("relative_azimuth", len(coords["relative_azimuth"]))
    nc.createDimension("water", len(coords["water"]))
    nc.createDimension("ozone", len(coords["ozone"]))
    nc.createDimension("target_pressure", len(coords["target_pressure"]))
    nc.createDimension("sensor_altitude", len(coords["sensor_altitude"]))
    nc.createDimension("wavelength", len(coords["wavelength"]))

    sol_zen_nc = nc.createVariable("sol_zen", "f4", ("sol_zen",))
    sol_zen_nc.standard_name = "solar_zenith_angle"
    sol_zen_nc.units = "degree"
    sol_zen_nc[:] = coords["sol_zen"]

    view_zen_nc = nc.createVariable("view_zen", "f4", ("view_zen",))
    view_zen_nc.standard_name = "sensor_zenith_angle"
    view_zen_nc.units = "degree"
    view_zen_nc[:] = coords["view_zen"]

    relative_azimuth_nc = nc.createVariable(
        "relative_azimuth", "f4", ("relative_azimuth",)
    )
    relative_azimuth_nc.standard_name = (
        "angle_of_rotation_from_solar_azimuth_to_platform_azimuth"
    )
    relative_azimuth_nc.long_name = "relative_azimuth_angle"
    relative_azimuth_nc.units = "degree"
    relative_azimuth_nc[:] = coords["relative_azimuth"]

    water_nc = nc.createVariable("water", "f4", ("water",))
    water_nc.standard_name = "atmosphere_mass_content_of_water_vapor"
    water_nc.units = "g cm-2"
    water_nc[:] = coords["water"]

    ozone_nc = nc.createVariable("ozone", "f4", ("ozone",))
    ozone_nc.standard_name = "equivalent_thickness_at_stp_of_atmosphere_ozone_content"
    ozone_nc.units = "atm-cm"
    ozone_nc[:] = coords["ozone"]

    pressure_nc = nc.createVariable("target_pressure", "f4", ("target_pressure",))
    pressure_nc.units = "mbar"
    pressure_nc[:] = coords["target_pressure"]

    altitude_nc = nc.createVariable("sensor_altitude", "f4", ("sensor_altitude",))
    altitude_nc.standard_name = "altitude"
    altitude_nc.units = "km"
    altitude_nc[:] = coords["sensor_altitude"]

    # Create the coordinates variables
    wavelength_nc = nc.createVariable("wavelength", "f4", ("wavelength",))
    wavelength_nc.standard_name = "radiation_wavelength"
    wavelength_nc.units = "nm"
    wavelength_nc[:] = coords["wavelength"]

    # Create the data variables
    dimensions = (
        "sol_zen",
        "view_zen",
        "relative_azimuth",
        "water",
        "ozone",
        "target_pressure",
        "sensor_altitude",
        "wavelength",
    )

    nc.createVariable(
        "tgv", "f4", dimensions, compression=compression, complevel=complevel
    )
    nc.createVariable(
        "tgs", "f4", dimensions, compression=compression, complevel=complevel
    )

    return nc


# Function to run model and accumulate results
def run_model_and_accumulate(start, end, commands, tgv, tgs):
    global counter  # Declare counter as global

    for i in range(start, end):
        command = commands[i]
        process = subprocess.run(command, shell=True, capture_output=True)

        temp = json.loads(process.stdout)

        tgv[i] = float(temp["global_gas_trans_upward"])
        tgs[i] = float(temp["global_gas_trans_downward"])
        counter += 1

    return


# Declare the global counter
counter = 0


if __name__ == "__main__":
    # Function to generate cartesian product of input iterables
    def cartesian_product(dimensions):
        return list(itertools.product(*dimensions))

    # TODO, run only for the 20 node wavelength of 6S
    #  they use liner interpolation for the rest of the spectrun anyway
    # Define your dimensions here
    dimensions = [
        np.arange(0, 90, 50).tolist(),  # sun zenith
        np.arange(0, 70, 50).tolist(),  # view zenith
        np.arange(0, 180, 30).tolist(),  # relative azimuth
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],  # H2O g/cm2
        [
            0.0,
            0.25,
            0.3,
            0.5,
            0.8,
        ],  # Ozone cm-atm https://gml.noaa.gov/ozwv/dobson/papers/wmobro/ozone.html
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
        # ], # aot550
        [750.0, 1013.0, 1100.0],  # pressure at target mb
        [-0.5, -1, -3, -4],  # sensor altitude -km
        [
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
        ],
    ]

    combination = cartesian_product(dimensions)

    print("number of combination: ", len(combination))

    tgv = [0] * len(combination)
    tgs = [0] * len(combination)

    # Create commands
    commands = []
    for i in range(len(combination)):
        command = (
            'echo "\n0 # IGEOM\n'
            + f"{combination[i][0]} 0.0 {combination[i][1]} {combination[i][2]} 1 1 #sun_zenith sun_azimuth view_zenith view_azimuth month day\n"
            + "8 # IDATM no gas\n"
            + f"{combination[i][3]}\n"
            + f"{combination[i][4]}\n"
            + "0 # IAER maritime\n"
            + "-1 # visibility\n"
            + f"{combination[i][5]} # XPS pressure at terget\n"
            + f"{combination[i][6]} # XPP sensor altitude\n"
            + "-1.0 -1.0 # UH20 UO3 below sensor\n"
            + "-1.0 # taer550 below sensor\n"
            + "-1 # IWAVE monochromatic\n"
            + f"{combination[i][7]} # wavelength\n"
            + "0 # INHOMO\n"
            + "0 # IDIREC\n"
            + "0 # IGROUN 0 = rho\n"
            + "0 # surface reflectance\n"
            + '-1 # IRAPP no atmospheric correction\n" | /home/raphael/PycharmProjects/reverie/reverie/6S/6sV2.1/sixsV2.1'
        )
        commands.append(command)

    lut_dir = "/home/raphael/PycharmProjects/reverie/reverie/data/lut"

    output_dir = os.path.join(lut_dir, "output")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Determine the number of workers to use
    num_workers = os.cpu_count()

    print(f"Running on {num_workers} threads")

    # Calculate the number of iterations per worker
    iterations_per_worker = len(combination) // num_workers

    start_time = time.perf_counter()

    # Create a ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Start each worker
        futures = []
        for i in range(num_workers):
            # Calculate the start and end indices for this worker
            start = i * iterations_per_worker
            end = (
                start + iterations_per_worker
                if i != num_workers - 1
                else len(combination)
            )

            # Start the worker
            futures.append(
                executor.submit(
                    run_model_and_accumulate, start, end, commands, tgv, tgs
                )
            )

        while counter < len(combination):
            time.sleep(1)  # Sleep for a second

            now = time.perf_counter()
            elapsed = now - start_time

            if elapsed > 0:  # To avoid division by zero
                print(f"\r({counter}/{len(combination)}) | ", end="")

                iterations_per_second = counter / elapsed
                print(f"Iterations per second: {iterations_per_second}", end="")

                estimated_total_time = (
                    len(combination) - counter
                ) / iterations_per_second
                print(
                    f" | Estimated time upon completion: {format_estimated_time(estimated_total_time)}",
                    end="",
                    flush=True,
                )

        # Wait for all workers to finish
        concurrent.futures.wait(futures)

        now = time.perf_counter()
        elapsed = now - start_time
        print(f"Time elapsed: {elapsed}")

        coords = {
            "sol_zen": dimensions[0],
            "view_zen": dimensions[1],
            "relative_azimuth": dimensions[2],
            "water": dimensions[3],
            "ozone": dimensions[4],
            "target_pressure": dimensions[5],
            "sensor_altitude": dimensions[6],
            "wavelength": dimensions[7],
        }

    nc = create_gas_output_nc(os.path.join(output_dir, "test_gas.nc"), coords)

    # Get the shape of the dimensions
    shape = [len(dimension) for dimension in dimensions]

    tgs_reshaped = np.reshape(tgs, shape)
    tgv_reshaped = np.reshape(tgv, shape)

    # Write the results to the file
    nc.variables["tgs"][
        :,
        :,
        :,
        :,
        :,
        :,
        :,
        :,
    ] = tgs_reshaped

    nc.variables["tgv"][
        :,
        :,
        :,
        :,
        :,
        :,
        :,
        :,
    ] = tgv_reshaped

    nc.close()
