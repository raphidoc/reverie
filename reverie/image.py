# Standard library imports
import os
from abc import ABC, abstractmethod



# Third party imports
from osgeo import gdal
import netCDF4
from tqdm import tqdm
import numpy as np
import pyproj
import re
import pendulum

# REVERIE import
from reverie.utils import helper
from reverie.utils.tile import Tile


class Image(ABC):
    """
    Image abstract class is used as the base template for image data structure in REVERIE.
    This class is to be expanded by sensor specific class as in ../sensor/wise/image.py

        Attributes
    ----------
    Affine : Affine()
        Affine trnasformation
    n_rows : int
        number of rows
    n_cols : int
        number of columns
    CRS : pyproj.crs.CRS
        a pyproj.crs.CRS object

    acq_time_z: Acquisition time in UTC
    acq_time_local: Acquisition time in local time
    central_lon_local_timezone: Used to compute solar geometry

    solar_zenith:
        spatially resolved solar zenith [?], is this really needed ? over a decakilometer scale difference is minime
    solar_azimuth: spatially resolved solar azimuth [?]

    viewing_zenith: spatially resolved viewing zenith [?]
    view_azimuth: spatially resolved viewing azimuth [?]
    relative_azimuth:spatially resolved relative azimuth (solar_azimuth - view_azimuth) [?]
    SampleIndex: position of pixel on the detector array spatial dimension (pushbroom sensor only)

        Methods
    -------
    cal_coordinate(): Compute the x, y projected coordinate from Affine, n_rows, n_cols and CRS
    """

    def __init__(self):
        self.declaration = "I belong to Image"

        # Geographic attributes
        self.Affine, self.n_rows, self.n_cols, self.CRS = None, None, None, None

        self.x, self.y, self.lon, self.lat = None, None, None, None
        self.central_lon, self.central_lat = None, None

        # Time attributes
        self.acq_time_z, self.acq_time_local = None, None
        self.central_lon_local_timezone = None

        # Geometric attributes
        self.solar_zenith = None
        self.solar_azimuth = None

        self.viewing_zenith = None
        self.view_azimuth = None
        self.relative_azimuth = None
        self.SampleIndex = None

    def cal_coordinate(self, affine, n_rows, n_cols, crs):
        """
        Compute the pixel coordinates

        Parameters
        ----------
        affine: Affine transformation
        n_rows: Number of rows
        n_cols: Number of column
        crs: pyproj CRS object

        Returns
        -------
        x: projected pixel coordinates
        y: projected pixel coordinates
        longitude:
        latitude:
        center_longitude:
        center_latitude:
        """
        # Define image coordinates
        cor_x, cor_y = helper.transform_xy(affine, rows=list(range(n_rows)), cols=list(range(n_cols)))
        x, y = cor_x, cor_y

        # transform the coordinates to WGS84 (EPSG:4326), extract longitude and latitude
        transformer = pyproj.Transformer.from_crs(crs.to_epsg(), 4326)
        xv, yv = np.meshgrid(x, y)
        lat, lon = transformer.transform(xv, yv)
        central_lon = lon[n_rows // 2, n_cols // 2]
        central_lat = lat[n_rows // 2, n_cols // 2]

        self.x, self.y, self.lon, self.lat, self.central_lon, self.central_lat = x, y, lon, lat, central_lon, central_lat

        #return x, y, lon, lat, central_lon, central_lat

    def cal_time(self, central_lon, central_lat):
        """
        Compute local time of image acquisition

        Parameters
        ----------
        central_lon: central longitude
        central_lat: central latitude

        Returns
        -------
        x: projected pixel coordinates
        y: projected pixel coordinates
        longitude:
        latitude:
        center_longitude:
        center_latitude:
        """
        # find the local timezone from the central longitude and latitude
        tz = helper.findLocalTimeZone(central_lon, central_lat)

        self.acq_time_local = tz.convert(self.acq_time_z)
        print(f"acquired UTC time:{self.acq_time_z}, and local time：{self.acq_time_local}")

        offset_hours = self.acq_time_local.offset_hours
        self.central_lon_local_timezone = offset_hours * 15
        print(f"central longitude of {tz}:{self.central_lon_local_timezone}")

    def cal_sun_geom(self):
        '''
        calculate solar zenith angle and azimuth angle
        :return:
        '''
        hourAngle = helper.calSolarHourAngel(self.lon, self.central_lon_local_timezone, self.acq_time_local)

        ## year, month and day
        year, month, day = self.acq_time_z.year, self.acq_time_z.month, self.acq_time_z.day

        declination = helper.calDeclination(year, month, day)
        self.solar_zenith, self.solar_azimuth = helper.calSolarZenithAzimuth(self.lat, hourAngle, declination)

        self.solar_zenith[~self.get_valid_mask()] = np.nan
        self.solar_azimuth[~self.get_valid_mask()] = np.nan

    def get_valid_mask(self, tile: Tile = None):
        if self._valid_mask is None:
            self.cal_valid_mask()
        if tile is None:
            return self._valid_mask
        else:
            return self._valid_mask[tile.sline:tile.eline,tile.spixl:tile.epixl]

    def cal_valid_mask(self):
        if self._valid_mask is None:
            iband = 0 #if self.__red_band_index is None else self.__red_band_index
            Lt = self.read_band(iband)
            self._valid_mask = (Lt > 0)

    def read_band(self, bandindex, tile: Tile = None):
        '''
        read DN for a given band
        :param bandindex: bandindex starts with 0
        :param tile: an instance of the Tile class used for tiled processing
        :return: re
        '''
        band_temp = self.src_ds.GetRasterBand(bandindex + 1)
        if tile:
            Lt = band_temp.ReadAsArray(
                xoff=tile[2],
                yoff=tile[0],
                win_xsize=tile.xsize,
                win_ysize=tile.ysize)
        else:
            Lt = band_temp.ReadAsArray()

        return Lt

    def cal_relative_azimuth(self):
        '''
        calculate relative azimuth angle
        :return:
        '''
        self.raa = np.abs(self.view_azimuth - self.solar_azimuth)


    #@abstractmethod
    def cal_view_geom(self):
        pass

    def to_netcdf(self):
        """
        Export Image to CF compliant NetCDF dataset
        :return: None
        """
        pass
