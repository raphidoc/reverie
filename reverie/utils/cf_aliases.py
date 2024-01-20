# TODO: get the standard_name of all the quantity that we could write from optical imagery ?
#  https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html

# standard name

# Radiometric quantities:
# There is no good standard name for water leaving reflectance

# Rrs = surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air [sr-1]
# F0 = solar_irradiance_per_unit_wavelength [W m-2 m-1]
# Radiance at sensor = upwelling_radiance_per_unit_wavelength_in_air [W m-2 m-1 sr-1]
# Wavelength = sensor_band_central_radiation_wavelength [m]

# Domain axis
# altitude: Altitude is the (geometric) height above the geoid, which is the reference geopotential surface. The geoid is similar to mean sea level. [m]
# latitude: Latitude is positive northward; its units of degree_north (or equivalent) indicate this explicitly. [degree_north]
# longitude: Longitude is positive eastward; its units of degree_east (or equivalent) indicate this explicitly [degree_east]
# time [s]
# projection_x_coordinate [m]
# projection_y_coordinate [m]


import requests
import xmltodict
from xml.dom.minidom import parse, parseString, Node
import untangle
import xml.etree.cElementTree as ET


def get_cf_std_name(url=None, alias: str = 'Lt'):

    std_name = None
    std_unit = None

    # Template
    # if alias in ['']:
    #     std_name = ''
    #     std_unit = ''

    # Radiometric quantities
    if alias in ['Lt']:
        std_name = 'upwelling_radiance_per_unit_wavelength_in_air'
        std_unit = 'W cm-2 nm-1 sr-1'

    # No standard
    if alias in ['rhot']:
        std_name = 'at_sensor_reflectance'
        std_unit = '1'

    # No standard
    if alias in ['rhow']:
        std_name = 'water_leaving_reflectance'
        std_unit = '1'

    if alias in ['Rrs']:
        std_name = 'surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air'
        std_unit = 'sr-1'

    if alias in ['F0']:
        std_name = 'solar_irradiance_per_unit_wavelength'
        std_unit = 'W cm-2 nm-1'

    # Geometric quantities:
    # solar_azimuth_angle [degree]
    # solar_zenith_angle [degree]

    # sensor_azimuth_angle [degree]
    # sensor_zenith_angle [degree]

    # Relative azimuth angle = angle_of_rotation_from_solar_azimuth_to_platform_azimuth [degree]
    # platform_azimuth_angle / sensor_azimuth_angle / relative_platform_azimuth_angle / relative_sensor_azimuth_angle ?

    if alias in ['SolZen']:
        std_name = 'solar_zenith_angle'
        std_unit = 'degree'

    if alias in ['SolAzm']:
        std_name = 'solar_azimuth_angle'
        std_unit = 'degree'

    if alias in ['ViewZen']:
        std_name = 'sensor_zenith_angle'
        std_unit = 'degree'

    if alias in ['ViewAzm']:
        std_name = 'sensor_azimuth_angle'
        std_unit = 'degree'

    if alias in ['RelativeAzimuth']:
        # Should be: angle_of_rotation_from_solar_azimuth_to_sensor_azimuth
        std_name = 'angle_of_rotation_from_solar_azimuth_to_platform_azimuth'
        std_unit = 'degree'

    # No standard
    if alias in ['SampleIndex']:
        std_name = 'position_index_of_across_track_push_broom_samples) '
        std_unit = ''

    if std_name is None:
        raise Exception(f"No match for {alias}")

    # TODO: automate retrieval of standard from the XML table at:
    #  http://cfconventions.org/Data/cf-standard-names/current/src/cf-standard-name-table.xml

    # response = requests.get(url)
    # tree = ET.ElementTree(ET.fromstring(response.content))
    # # test = tree.findall('entry')
    # qry=f"entry[@id='{std_name}']"
    # test = tree.find(qry)
    # dic = {}
    # for x in test.iter():
    #     print(x.text)

    return std_name, std_unit



