"""
This module intend to provide a fast and efficient way of running 6S for LUT generation in python.
"""

import subprocess


def run_6s_gas_lut(**kwargs):
    # Parse kwargs to generate the 6S input file
    input_str = "\n".join(
        [
            # Geometrical conditions
            "0 # Geometrical conditions igeom",
            "%(sol_zen).1f %(sol_azi).1f %(view_zen).1f %(view_azi).1f %(month)s %(day)s"
            % kwargs,
            # Atmospheric conditions, gas
            "8 # Atmospheric conditions, gas (idatm)",
            "%(water_vapor).2f %(ozone).2f" % kwargs,
            # Aerosol type and concentration
            "2 #iaer",
            "%(visibility)s # Visibility (km)" % kwargs,
            "%(aot550).2f" % kwargs,
            # Target altitude as pressure in mb
            "%(target_altitude).2f # target altitude (xps)" % kwargs,
            # Sensor altitude in km relative to target altitude
            "%(sensor_altitude).2f # Sensor altitude (xpp)" % kwargs,
            # If aircraft simulation (> -110 < 0) enter water vapor, ozone content
            # Aircraft water vapor, ozone and aot550
            "%(aircraft_water_vapor).2f %(aircraft_ozone).2f" % kwargs,
            "%(aircraft_aot550).2f" % kwargs,
            # Spectrum selection for simulation
            "25 # Spectrum selection (iwave)",
            # "%(wavelength).3f" % kwargs,
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

    # with open("/home/raphael/PycharmProjects/reverie/reverie/6S/in.txt", "w") as f:
    #     f.writelines(input_str)

    # Run the 6S executable with the input parameters
    exec_6sv2 = "/home/raphael/PycharmProjects/reverie/reverie/6S/6sV2.1/sixsV2.1"
    process = subprocess.run(exec_6sv2, input=input_str, text=True, capture_output=True)

    process.stderr
    process.stdout

    # Check if the process completed successfully
    if process.returncode != 0:
        raise Exception("6S execution failed with error code %s" % process.returncode)


if __name__ == "__main__":
    sixs_kwargs = {
        # Geometrical conditions (igeom: 0, 1, 2, 3, 4, 5, 6, 7)
        "user_geometry": int,
        # If user_geometry is selected, then enter:
        "sol_zen": 0.0,
        "sol_azi": 0.0,
        "view_zen": 0.0,
        "view_azi": 0.0,
        "month": 1,
        "day": 1,
        # Atmospheric conditions, gas (idatm: 0, 1, 2, 3, 4, 5, 6) REDUCE PARAMS
        "no_gas": int,
        "tropical": 1,
        "mid_lat_summer": 2,
        "mid_lat_winter": 3,
        "sub_arctic_summer": 4,
        "sub_arctic_winter": 5,
        "us_standard_62": 6,
        "user_water_vapor_ozone": 8,
        # If user_water_vapor_ozone is selected, then enter:
        "water_vapor": 0,
        "ozone": 0,
        # Aerosol type and profile (iaer: -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
        "no_aerosols": 0,
        "continental": 1,
        "maritime": 2,
        "urban": 3,
        "desert": 5,
        "biomass_burning": 6,
        "stratospheric": 7,
        # If default aerosol model is selected (1, 2, 3, 5, 6, 7) then either enter:
        # WARNING: if no_aerosols is selected, visibility must be set to -1
        "visibility": 0,  # km
        # Or:
        "aot550": 0.3,
        "user_aerosol": 4,
        # If user_aerosol is selected, then enter:
        "user_aerosol_parameters": {
            # Percentage (0-1) of:
            "dust_like": 0,
            "water_soluble": 0,
            "oceanic": 0,
            "soot": 0,
        },
        # Target altitude (xps)
        # if >= 0 then expressed in mb, if < 0 then expressed in km
        "target_altitude": 0,
        # Sensor altitude (xpp)
        # -1000 = satellite
        # 0 = ground level
        # -100< sensor_altitude <0 = aircraft simulation, altitude in km relative to target_altitude
        "sensor_altitude": -1,
        # If aircraft simulation enter water vapor, ozone content:
        # If Unknown enter negative values
        "aircraft_water_vapor": -1,
        "aircraft_ozone": -1,
        # Enter aerosol optical thickness at 550 nm on next line:
        "aircraft_aot550": -1,
        # Spectrum selection for simulation (iwave: -1, 0, 1) # REDUCE PARAMS
        "monochromatic": -1,
        # If monochromatic is selected, then enter:
        "wavelength": 0.443,  # in um
        "flat_rsr": 0,
        "custom_rsr": 0,
        # If either flat_rsr or custom_rsr is selected, then enter:
        "wavelength_inf": 0,  # in um
        "wavelength_sup": 0,  # in um
        # If custom_rsr is selected, then enter:
        "filter_function": None,  # By step of 0.0025 um
        # Ground reflectance (inhomo: 0) REDUCE PARAMS
        "homogeneous": 0,
        # If homogeneous is selected, then enter either:
        "no_directional_effects": 0,
        # If directional_effects is selected, then enter:
        "constant_rho": 0,
        # Then you enter the constant value of the ground reflectance
        "constant_rho_value": 0,
        "directional_effects": 1,
        # Atmospheric correction (irapp: -1, 0, 1)
        # if -1 then no atmospheric correction
        # if atmospheric_correction > 0 then assume input radiance in W m-2 sr-1 um-1
        # if -1 < atmospheric_correction < 0 then assume input reflectance
        "atmospheric_correction": -1,
    }

    # from Py6S import *
    #
    # # Instance the class
    # s = SixS("/home/raphael/PycharmProjects/6SV-1.1/6SV1.1/sixsV1.1")
    # # Define an atmosphere with no aerosol effects and ground as non-reflecting
    # s.aero_profile = AeroProfile.NoAerosols
    # s.ground_reflectance = GroundReflectance.HomogeneousLambertian(0)
    #
    # # Atmospheric profile (Gas)
    # s.atmos_profile = AtmosProfile.UserWaterAndOzone(0, 0)
    #
    # # Geometry
    # # By keeping solar_a at 0 we are assuming that the sun at the North,
    # # therefore effectively computing for relative azimuth angle.
    # s.geometry = Geometry.User()
    # s.geometry.solar_z = 0
    # s.geometry.solar_a = 0
    # s.geometry.view_z = 0
    # s.geometry.view_a = 0
    #
    # # Altitude
    # s.altitudes = Altitudes()
    # s.altitudes.set_target_pressure(0)
    # s.altitudes.set_sensor_custom_altitude(0)

    run_6s_gas_lut(**sixs_kwargs)
