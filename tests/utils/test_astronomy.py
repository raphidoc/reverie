import pytest
import datetime

from reverie.utils import astronomy


def test_sun_geom_noaa():
    """Test sun_geom_noaa function.
    Notes
    -----
    The testing values are from the function intself.
    I checked those for near equality with the values from the NOAA spreadsheet.
    """
    date_time = datetime.datetime(2019, 8, 18, 17, 35, 29)
    time_zone = -4
    obs_latitude = 49.077464
    obs_longitude = -68.323974

    assert astronomy.sun_geom_noaa(
        date_time, time_zone, obs_latitude, obs_longitude
    ) == (
        267.0648595143452,
        70.1264503568939,
    ), "Should be azimuth  ~ 267.0648595143452 and zenith ~ 70.1264503568939"
