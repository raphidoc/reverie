import math

import netCDF4 as nc

from reverie import ReveCube
from reverie.ancillary import oceandata


def add_ancillary(bundle_path: str = None, sensor: str = None):
    # l1_convert(bundle_path)

    l1 = ReveCube.from_reve_nc(
        in_path="/D/Data/WISE/MC-10A/190820_MC-10A-WI-1x1x1_v02-L1G.nc"
    )

    dt = l1.acq_time_z
    central_lon = l1.lon[round(len(l1.lon) / 2)]
    central_lat = l1.lat[round(len(l1.lat) / 2)]
    print(
        f"Getting ancillary data for {dt.strftime('%Y-%m-%d %H:%M:%SZ')} {central_lon:.3f}E {central_lat:.3f}N"
    )

    anc = oceandata.get(dt, central_lon, central_lat, local_dir="/D/Data/TEST/anc")

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

    wind_direction = l1.in_ds.createVariable("wind_direction", "f4")
    wind_direction.standard_name = "wind_direction"
    wind_direction.units = "degrees"
    wind_direction.long_name = "Wind direction at 10m"
    wind_direction.description = "GMAO MERRA-2"
    wind_direction[:] = math.atan2(anc.variables["U10M"], anc.variables["V10M"]) * 180 / math.pi

    surface_air_pressure = l1.in_ds.createVariable("surface_air_pressure", "f4")
    surface_air_pressure.standard_name = "surface_air_pressure"
    surface_air_pressure.units = "mb"
    surface_air_pressure.long_name = "Surface air pressure"
    surface_air_pressure.description = "GMAO MERRA-2"
    surface_air_pressure[:] = anc.variables["PS"] / 100  # convert [Pa] to [mb]

    water_vapor = l1.in_ds.createVariable("atmosphere_mass_content_of_water_vapor", "f4")
    water_vapor.standard_name = "atmosphere_mass_content_of_water_vapor"
    water_vapor.units = "g cm-2"
    water_vapor.long_name = "Atmosphere mass content of water vapor"
    water_vapor.description = "GMAO MERRA-2"
    water_vapor[:] = anc.variables["TQV"] / 10  # convert [kg m-2] to [g cm-2]

    ozone = l1.in_ds.createVariable("equivalent_thickness_at_stp_of_atmosphere_ozone_content", "f4")
    ozone.standard_name = "equivalent_thickness_at_stp_of_atmosphere_ozone_content"
    ozone.units = "cm-atm"
    ozone.long_name = "Equivalent thickness at STP of atmosphere ozone content"
    ozone.description = "GMAO MERRA-2"
    ozone[:] = anc.variables["TO3"] / 1000  # convert [Dobson] to [cm-atm]

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
    ac_run("/D/Data/TEST/TEST_WISE/MC-50/")
