import math
import os
import numpy as np
import logging

import netCDF4 as nc
from itertools import compress


from reverie import ReveCube
from reverie.ancillary import obpg


def add_ancillary(l1: ReveCube):

    # test if ancillary data is already present
    required_anc = ["wind_speed", "wind_direction", "surface_air_pressure",
                     "atmosphere_mass_content_of_water_vapor",
                     "equivalent_thickness_at_stp_of_atmosphere_ozone_content",
                     "aerosol_optical_thickness_at_555_nm"]
    existing_anc = set(l1.in_ds.variables.keys())
    if all(var in existing_anc for var in required_anc):
        logging.info("Ancillary data already present.")
        return

    anc = obpg.get_gmao(l1)

    # Test for empty values returned in anc object
    test_list = [math.isnan(anc[key].values) for key in anc.variables]

    missing = list(compress(anc.variables, test_list))

    ancillary = [
    "U10M",
    "V10M",
    "PS",
    "TQV",
    "TO3",
    "TOTEXTTAU"
    ]

    test = [miss in ancillary for miss in missing]

    if any(test):
        list(compress(anc.variables, test_list))
        raise "Missing ancillary data " + str(missing)

    # Wind speed and direction from http://colaweb.gmu.edu/dev/clim301/lectures/wind/wind-uv
    # Ozone conversion from https://www.ldeo.columbia.edu/users/menke/envdata/ozone/oz_lec1.html

    # Open the dataset with netcdf4 and write the ancilliary variable

    l1.in_ds.close()

    l1.in_ds = nc.Dataset(l1.in_path, "a", format="NETCDF4")

    wind_speed = l1.in_ds.createVariable("wind_speed", "f4")
    wind_speed.standard_name = "wind_speed"
    wind_speed.units = "m s-1"
    wind_speed.long_name = "Wind speed at 10m"
    wind_speed.description = "GMAO MERRA-2"
    wind_speed[:] = math.sqrt(anc.variables["U10M"] ** 2 + anc.variables["V10M"] ** 2)
    # l1.in_ds["wind_speed"][:] = math.sqrt(anc.variables["U10M"] ** 2 + anc.variables["V10M"] ** 2)

    wind_direction = l1.in_ds.createVariable("wind_direction", "f4")
    wind_direction.standard_name = "wind_direction"
    wind_direction.units = "degrees"
    wind_direction.long_name = "Wind direction at 10m"
    wind_direction.description = "GMAO MERRA-2"
    wind_direction[:] = (
        math.atan2(anc.variables["U10M"], anc.variables["V10M"]) * 180 / math.pi
    )
    # l1.in_ds["wind_direction"][:] = (
    #         math.atan2(anc.variables["U10M"], anc.variables["V10M"]) * 180 / math.pi
    #     )

    surface_air_pressure = l1.in_ds.createVariable("surface_air_pressure", "f4")
    surface_air_pressure.standard_name = "surface_air_pressure"
    surface_air_pressure.units = "mb"
    surface_air_pressure.long_name = "Surface air pressure"
    surface_air_pressure.description = "GMAO MERRA-2"
    surface_air_pressure[:] = anc.variables["PS"] / 100  # convert [Pa] to [mb]
    # l1.in_ds["surface_air_pressure"][:] = anc.variables["PS"] / 100

    water_vapor = l1.in_ds.createVariable(
        "atmosphere_mass_content_of_water_vapor", "f4"
    )
    water_vapor.standard_name = "atmosphere_mass_content_of_water_vapor"
    water_vapor.units = "g cm-2"
    water_vapor.long_name = "Atmosphere mass content of water vapor"
    water_vapor.description = "GMAO MERRA-2"
    water_vapor[:] = anc.variables["TQV"] / 10  # convert [kg m-2] to [g cm-2]
    # l1.in_ds["atmosphere_mass_content_of_water_vapor"][:] = anc.variables["TQV"] / 10

    ozone = l1.in_ds.createVariable(
        "equivalent_thickness_at_stp_of_atmosphere_ozone_content", "f4"
    )
    ozone.standard_name = "equivalent_thickness_at_stp_of_atmosphere_ozone_content"
    ozone.units = "cm-atm"
    ozone.long_name = "Equivalent thickness at STP of atmosphere ozone content"
    ozone.description = "GMAO MERRA-2"
    ozone[:] = anc.variables["TO3"] / 1000  # convert [Dobson] to [cm-atm]
    # l1.in_ds["equivalent_thickness_at_stp_of_atmosphere_ozone_content"][:] = anc.variables["TO3"] / 1000

    aot = l1.in_ds.createVariable("aerosol_optical_thickness_at_555_nm", "f4")
    aot.standard_name = "aerosol_optical_thickness_at_555_nm"
    aot.units = "1"
    aot.long_name = "aerosol_optical_thickness_at_555_nm"
    aot.description = "GMAO MERRA-2 TOTEXTTAU"
    aot[:] = anc.variables["TOTEXTTAU"]
    # l1.in_ds["aerosol_optical_thickness_at_555_nm"][:] = anc.variables["TOTEXTTAU"]

    l1.in_ds.close()

    # test = l1.in_ds.assign(
    #     {
    #         "wind_speed": math.sqrt(anc.variables["U10M"] ** 2 + anc.variables["V10M"] ** 2),  # in [m s-1]
    #         "wind_direction": math.atan2(anc.variables["U10M"], anc.variables["V10M"]) * 180 / math.pi,
    #         "surface_pressure": anc.variables["PS"] / 100,  # convert [Pa] to [mb]
    #         "water_vapor": anc.variables["TQV"] / 10,  # convert [kg m-2] to [g cm-2]
    #         "ozone": anc.variables["TO3"] / 1000, # convert [Dobson] to [cm-atm]
    #     }
    # )

if __name__ == "__main__":
    image_dir = "/D/Data/WISE/"

    images = [
        # "ACI-10A/220705_ACI-10A-WI-1x1x1_v01-L1CG.nc",
        # "ACI-11A/220705_ACI-11A-WI-1x1x1_v01-L1CG.nc",
        # "ACI-12A/220705_ACI-12A-WI-1x1x1_v01-L1CG.nc",
        # "ACI-13A/220705_ACI-13A-WI-1x1x1_v01-L1CG.nc",
        "ACI-14A/220705_ACI-14A-WI-1x1x1_v01-L1CG.nc"
        # "MC-50A/190818_MC-50A-WI-2x1x1_v02-L1CG.nc",
        # "MC-37A/190818_MC-37A-WI-1x1x1_v02-L1CG.nc",
        # "MC-10A/190820_MC-10A-WI-1x1x1_v02-L1G.nc",
    ]

    for image in images:
        l1 = ReveCube.from_reve_nc(os.path.join(image_dir, image))

        add_ancillary(l1)
