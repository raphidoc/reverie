from math import *
import datetime


def date_time_to_julian_day(date_time: datetime):
    """Convert a datetime to a Julian Day
    Parameters
    ----------
    date_time: datetime
        The date time to convert to Julian Day

    Returns
    -------
    float
        The Julian Day

    Notes
    -----
    The Julian Day is the continuous count of days since the beginning of the Julian Period used primarily by astronomers.
    The algorithm used here is eq. 7.1 from [1]_ and is valid for any date in the Gregorian calendar.

    References
    ----------
    .. [1] Meeus, J. (1998) Astronomical algorithms. 2nd ed. Richmond, Va: Willmann-Bell.
    """

    # TODO: not sure about the meaning of INT() in Meeus (1998) is it int or trunc ?
    year = date_time.year
    month = date_time.month
    day = date_time.day

    decimal_hour = date_time.hour / 24
    decimal_minute = date_time.minute / 1440
    decimal_second = date_time.second / 86400

    day = day + decimal_hour + decimal_minute + decimal_second

    if month <= 2:
        year = year - 1
        month = month + 12

    a = int(year / 100)
    b = 2 - a + int(a / 4)

    julian_day = (
        int(365.25 * (year + 4716)) + int(30.61 * (month + 1)) + day + b - 1524.5
    )

    return julian_day


def sun_geom_meeus(date_time, obs_latitude):
    julian_day = date_time_to_julian_day(date_time)

    julian_centuries = (julian_day - 2451545.0) / 36525

    mean_lon_sun = (
        280.46645 + 36000.76983 * julian_centuries + 0.0003032 * julian_centuries**2
    )

    # In nooa calc, it is mean_lon_sun % 360
    mean_lon_sun = mean_lon_sun % 360

    # Mean longitude of the moon: $L^'$
    mean_lon_moon = 218.3165 + 481267.8813 * julian_centuries

    # Mean anomaly of the sun: $M$ [degree]
    mean_anomaly_sun = (
        357.52910
        + 35999.05030 * julian_centuries
        - 0.0001559 * julian_centuries**2
        - 0.00000048 * julian_centuries**3
    )

    # Eccentricity of the Earth's orbit: $e$
    eccentricity_earth = (
        0.016708617
        - 0.000042037 * julian_centuries
        - 0.0000001236 * julian_centuries**2
    )

    # Sun equation of center: $C$
    center_sun = (
        (1.914600 - 0.004817 * julian_centuries - 0.000014 * julian_centuries**2)
        * sin(radians(mean_anomaly_sun))
        + (0.019993 - 0.000101 * julian_centuries) * sin(2 * radians(mean_anomaly_sun))
        + 0.000290 * sin(3 * radians(mean_anomaly_sun))
    )

    # Sun true longitude: $\theta$ [degree]
    true_longitude_sun = mean_lon_sun + center_sun

    # Sun true anomaly: $v$ [degree]
    true_anomaly_sun = mean_anomaly_sun + center_sun

    # Sun radius vector (sun-earth distance): $R$ [astronomical units]
    radius_vector_sun = (1.000001018 * (1 - eccentricity_earth**2)) / (
        1 + eccentricity_earth * cos(radians(true_anomaly_sun))
    )

    # Longitude of the ascending node of the moon mean orbit: $\omega$ [degree]
    # Simplified polynomial for the reduced precision nutation obliquity
    # See page 132 of Meeus (1998)
    # Also called as a correction factor for nutation and aberration on page 152
    lon_asc_node_moon_ecliptic = (
        125.04452 - 1934.136261 * julian_centuries
    )  # + 0.0020708 * julian_centuries**2 + julian_centuries**3/450000

    # Sun apparent longitude: $\lambda$
    apparent_longitude_sun = (
        true_longitude_sun
        - 0.00569
        - 0.00478 * sin(radians(lon_asc_node_moon_ecliptic))
    )

    nutation_longitude = (
        -17.20 * sin(radians(lon_asc_node_moon_ecliptic))
        - 1.32 * sin(2 * radians(mean_lon_sun))
        - 0.23 * sin(2 * radians(mean_lon_moon))
        + 0.21 * sin(2 * radians(lon_asc_node_moon_ecliptic))
    )

    # Nutation in obliquity: $\delta\epsilon$
    # Reduced precision at 0".1
    nutation_obliquity = (
        9.20 * cos(radians(lon_asc_node_moon_ecliptic))
        + 0.57 * cos(2 * radians(mean_lon_sun))
        + 0.10 * cos(2 * radians(mean_lon_moon))
        - 0.09 * cos(2 * radians(lon_asc_node_moon_ecliptic))
    )

    # Mean obliquity of the ecliptic: $\epsilon_o$
    # The input value is given as 23° 26' 21".448
    # TODO: different values for the two formulas that should be the same.
    # mean_obliquity_ecliptic = (
    #     (23 + (26 / 60) + (21.448 / 3600))
    #     - 46.8150 * julian_centuries
    #     - 0.00059 * julian_centuries**2
    #     + 0.001813 * julian_centuries**3
    # )

    # Fromm noaa
    mean_obliquity_ecliptic = (
        23
        + (
            26
            + (
                (
                    21.448
                    - julian_centuries
                    * (
                        46.815
                        + julian_centuries * (0.00059 - julian_centuries * 0.001813)
                    )
                )
            )
            / 60
        )
        / 60
    )

    # Truee obliquity of the ecliptic: $\epsilon$
    true_obliquity_ecliptic = mean_obliquity_ecliptic + nutation_obliquity

    mean_obliquity_ecliptic_cor = mean_obliquity_ecliptic
    +0.00256 * cos(radians(lon_asc_node_moon_ecliptic))

    # Right ascension of the sun: $a$
    # + 0.00256 * math.cos(omega) is the correction factor when computing apparent position
    right_ascension_sun = atan2(
        cos(radians(mean_obliquity_ecliptic_cor))
        * sin(radians(apparent_longitude_sun)),
        cos(radians(apparent_longitude_sun)),
    )

    right_ascension_sun = degrees(right_ascension_sun)

    # Declination of the sun
    declination_sun = asin(
        sin(radians(mean_obliquity_ecliptic_cor)) * sin(radians(apparent_longitude_sun))
    )

    declination_sun = degrees(declination_sun)

    # Greenwich mean sideral time: $\theta_o$
    greenwich_mean_sideral_time = (
        280.46061837
        + 360.9864736629 * (julian_day - 2451545.0)
        + 0.000387933 * julian_centuries**2
        - julian_centuries**3 / 38710000
    )

    # nutation in right ascension
    nutation_right_ascension = nutation_longitude * cos(
        radians(true_obliquity_ecliptic)
    )

    # Greenwich mean sideral time corrected for nutation
    greenwich_mean_sideral_time = greenwich_mean_sideral_time + nutation_right_ascension

    # Local hour angle: $H$
    local_hour_angle = greenwich_mean_sideral_time - obs_latitude - right_ascension_sun

    # Sun azimuth: $A$
    # Positive westward from south
    sun_azimuth = atan2(
        sin(local_hour_angle),
        cos(local_hour_angle) * sin(obs_latitude)
        - tan(declination_sun) * cos(obs_latitude),
    )

    # Sun altitude: $h$
    # Positive above the horizon
    sun_altitude = sin(obs_latitude) * sin(declination_sun) + cos(obs_latitude) * cos(
        declination_sun
    ) * cos(local_hour_angle)

    sun_azimuth = degrees(sun_azimuth)
    sun_altitude = degrees(sun_altitude)

    return sun_azimuth, sun_altitude


def sun_geom_noaa(date_time, time_zone, obs_latitude, obs_longitude):
    julian_day = date_time_to_julian_day(date_time)

    julian_centuries = (julian_day - 2451545.0) / 36525

    mean_lon_sun = (
        280.46645 + 36000.76983 * julian_centuries + 0.0003032 * julian_centuries**2
    ) % 360

    # Mean anomaly of the sun: $M$ [degree]
    mean_anomaly_sun = (
        357.52910
        + 35999.05030 * julian_centuries
        - 0.0001559 * julian_centuries**2
        - 0.00000048 * julian_centuries**3
    )

    # Eccentricity of the Earth's orbit: $e$
    eccentricity_earth = (
        0.016708617
        - 0.000042037 * julian_centuries
        - 0.0000001236 * julian_centuries**2
    )

    # Sun equation of center: $C$
    center_sun = (
        (1.914600 - 0.004817 * julian_centuries - 0.000014 * julian_centuries**2)
        * sin(radians(mean_anomaly_sun))
        + (0.019993 - 0.000101 * julian_centuries) * sin(2 * radians(mean_anomaly_sun))
        + 0.000290 * sin(3 * radians(mean_anomaly_sun))
    )

    # Sun true longitude: $\theta$ [degree]
    true_longitude_sun = mean_lon_sun + center_sun

    # Sun true anomaly: $v$ [degree]
    true_anomaly_sun = mean_anomaly_sun + center_sun

    # Longitude of the ascending node of the moon mean orbit: $\omega$ [degree]
    # Simplified polynomial for the reduced precision nutation obliquity
    # See page 132 of Meeus (1998)
    # Also called as a correction factor for nutation and aberration on page 152
    lon_asc_node_moon_ecliptic = (
        125.04452 - 1934.136261 * julian_centuries
    )  # + 0.0020708 * julian_centuries**2 + julian_centuries**3/450000

    # Sun apparent longitude: $\lambda$
    apparent_longitude_sun = (
        true_longitude_sun
        - 0.00569
        - 0.00478 * sin(radians(lon_asc_node_moon_ecliptic))
    )

    # Fromm noaa
    mean_obliquity_ecliptic = (
        23
        + (
            26
            + (
                (
                    21.448
                    - julian_centuries
                    * (
                        46.815
                        + julian_centuries * (0.00059 - julian_centuries * 0.001813)
                    )
                )
            )
            / 60
        )
        / 60
    )

    # Mean obliquity of the ecliptic [degree]
    mean_obliquity_ecliptic_cor = mean_obliquity_ecliptic
    +0.00256 * cos(radians(lon_asc_node_moon_ecliptic))

    # Right ascension of the sun: $a$ [degree]
    right_ascension_sun = degrees(
        atan2(
            cos(radians(mean_obliquity_ecliptic_cor))
            * sin(radians(apparent_longitude_sun)),
            cos(radians(apparent_longitude_sun)),
        )
    )

    # Declination of the sun
    declination_sun = degrees(
        asin(
            sin(radians(mean_obliquity_ecliptic_cor))
            * sin(radians(apparent_longitude_sun))
        )
    )

    var_y = tan(mean_obliquity_ecliptic_cor / 2 * pi / 180) * tan(
        mean_obliquity_ecliptic_cor / 2 * pi / 180
    )

    # Equation of time from NOAA, don't know where it comes from
    eq_of_time = 4 * degrees(
        var_y * sin(2 * radians(mean_lon_sun))
        - 2 * eccentricity_earth * sin(radians(mean_anomaly_sun))
        + 4
        * eccentricity_earth
        * var_y
        * sin(radians(mean_anomaly_sun))
        * cos(2 * radians(mean_lon_sun))
        - 0.5 * var_y * var_y * sin(4 * radians(mean_lon_sun))
        - 1.25
        * eccentricity_earth
        * eccentricity_earth
        * sin(2 * radians(mean_anomaly_sun))
    )

    decimal_day = (
        date_time.hour * 3600 + date_time.minute * 60 + date_time.second
    ) / 86400

    # True solar time from NOAA [minute]
    # In spreadsheet E is time in decimal day
    # B4 is longitude positive towards east
    # B5 is time zone, positive towards east
    true_solar_time = (
        decimal_day * 1440 + eq_of_time + 4 * obs_longitude - 60 * time_zone
    ) % 1440

    # Hour angle from NOAA [degree]
    if true_solar_time / 4 < 0:
        local_hour_angle = true_solar_time / 4 + 180
    else:
        local_hour_angle = true_solar_time / 4 - 180

    # Sun zenith angle from NOAA [deg]
    # B3 is latitude positive to north
    sun_zenith = degrees(
        acos(
            sin(radians(obs_latitude)) * sin(radians(declination_sun))
            + cos(radians(obs_latitude))
            * cos(radians(declination_sun))
            * cos(radians(local_hour_angle))
        )
    )

    # Sun azimuth from NOAA from [degree]
    if local_hour_angle > 0:
        sun_azimuth = (
            degrees(
                acos(
                    (
                        (sin(radians(obs_latitude)) * cos(radians(sun_zenith)))
                        - sin(radians(declination_sun))
                    )
                    / (cos(radians(obs_latitude)) * sin(radians(sun_zenith)))
                )
            )
            + 180
        ) % 360
    else:
        sun_azimuth = (
            540
            - degrees(
                acos(
                    (
                        (sin(radians(obs_latitude)) * cos(radians(sun_zenith)))
                        - sin(radians(declination_sun))
                    )
                    / (cos(radians(obs_latitude)) * sin(radians(sun_zenith)))
                )
            )
        ) % 360

    return sun_azimuth, sun_zenith


if __name__ == "__main__":
    obs_latitude = 49.077464
    obs_longitude = -68.323974
    time_zone = -4
    date_time = datetime.datetime(2019, 8, 18, 17, 35, 29)

    sun_geom_noaa(date_time, time_zone, obs_latitude, obs_longitude)

    #sun_geom_meeus(date_time, obs_latitude)
