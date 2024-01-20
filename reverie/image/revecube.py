# Standard library imports
import array
import os
from abc import ABC, abstractmethod
import datetime

# Third party imports
from osgeo import gdal
import netCDF4
import cfunits
from tqdm import tqdm
import numpy as np
import pyproj
import re
import pendulum

# REVERIE import
from reverie.utils import helper
from reverie.utils.tile import Tile
from reverie.utils.cf_aliases import get_cf_std_name


class ReveCube(ABC):
    """
    ReveCube abstract class is used as the base template for spectral imagery data structure in REVERIE.
    This class is to be expanded by sensor specific class as in ../sensor/wise/read_pix.py

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
    sample_index: position of pixel on the detector array spatial dimension (pushbroom sensor only)

        Methods
    -------
    cal_coordinate(): Compute the x, y projected coordinate from Affine, n_rows, n_cols and CRS
    """

    def __init__(self):

        # Datasets attribute
        self.src_ds = None
        self.net_ds = None
        self.no_data = None
        # TODO: scale factor should be handled variable specific
        #  it might not be appropriate to apply the same to radiometric and geometric data for example
        self.scale_factor = None
        self.proj_var = None

        # Radiometric attributes
        self.wavelength = None

        # Geographic attributes
        # Altitude (z) could alternatively belong the Geometric attributes ?
        self.z = None
        self.Affine, self.n_rows, self.n_cols, self.CRS = None, None, None, None

        self.x, self.y, self.lon, self.lat = None, None, None, None
        self.lon_grid, self.lat_grid = None, None
        self.center_lon, self.center_lat = None, None

        # Time attributes
        self.acq_time_z, self.acq_time_local = None, None
        self.central_lon_local_timezone = None

        # Geometric attributes
        self.solar_zenith = None
        self.solar_azimuth = None

        self.viewing_zenith = None
        self.view_azimuth = None
        self.relative_azimuth = None
        self.sample_index = None

        # Component attribute
        self.PixelExtractor = None

    def __str__(self):
        print("")


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
        # Define image coordinates from pixel number
        x, y = helper.transform_xy(affine, rows=list(range(n_rows)), cols=list(range(n_cols)))

        # transform the coordinates to WGS84 (EPSG:4326), extract longitude and latitude
        transformer = pyproj.Transformer.from_crs(crs.to_epsg(), 4326, always_xy=True)
        xv, yv = np.meshgrid(x, y)

        lon_grid, lat_grid = transformer.transform(xv, yv)

        lon, lat = lon_grid[0,:], lat_grid[:,0]

        #print(f"lat shape: {lat.shape}, lon shape: {lon.shape}")
        print(f"n_rows({n_rows}) = y({len(y)}) = lat({len(lat)})\r\nn_cols({n_cols}) = x({len(x)}) = lon({len(lon)})")

        # Could just use the lon lat vector
        central_lon = lon_grid[n_rows // 2, n_cols // 2]
        central_lat = lat_grid[n_rows // 2, n_cols // 2]

        self.lon_grid, self.lat_grid = lon_grid, lat_grid
        self.x, self.y, self.lon, self.lat, self.center_lon, self.center_lat = x, y, lon, lat, central_lon, central_lat

        #return x, y, lon, lat, center_lon, center_lat

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
        hourAngle = helper.calSolarHourAngel(self.lon_grid, self.central_lon_local_timezone, self.acq_time_local)

        ## year, month and day
        year, month, day = self.acq_time_z.year, self.acq_time_z.month, self.acq_time_z.day

        declination = helper.calDeclination(year, month, day)
        self.solar_zenith, self.solar_azimuth = helper.calSolarZenithAzimuth(self.lat_grid, hourAngle, declination)

        self.solar_zenith[~self.get_valid_mask()] = np.nan
        self.solar_azimuth[~self.get_valid_mask()] = np.nan

    def get_valid_mask(self, tile: Tile = None):
        """
        Get the valid mask for the enire image or a specific tile
        :param tile: an object of class tile.Tile()
        :return:
        """
        if self._valid_mask is None:
            self.cal_valid_mask()
        if tile is None:
            return self._valid_mask
        else:
            return self._valid_mask[tile.sline:tile.eline,tile.spixl:tile.epixl]

    def cal_valid_mask(self):
        """
        Calculate the mask of valid pixel for the entire image (!= nodata)
        :return: _valid_mask
        """
        if self._valid_mask is None:
            iband = 0
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
        self.relative_azimuth = np.abs(self.view_azimuth - self.solar_azimuth)

    #@abstractmethod
    def cal_view_geom(self):
        pass

    def create_dataset_nc(self, out_file: str = None):
        """Create CF-1.0 compliant NetCDF dataset from DateCube
        CF-1.0 (http://cfconventions.org/Data/cf-conventions/cf-conventions-1.0/build/cf-conventions.html#dimensions)
         is chosen to ensure compatibility with the GDAL NetCDF driver:
         (https://gdal.org/drivers/raster/netcdf.html).

        This format can also be read by SNAP but the spectrum view tool does not find any spectral dimension.
        Quinten wrote something about that in ACOLITE.

        :return: None
        """

        if out_file is None:
            raise Exception("out_file file not set, cannot create dataset")

        try:
            net_ds = netCDF4.Dataset(out_file, "w", format="NETCDF4")
        except Exception as e:
            print(e)
            return

        # TODO validate that it follow the convention with cfdm / cf-python. For compatibility with GDAL NetCDF driver use CF-1.0
        net_ds.Conventions = 'CF-1.0'
        net_ds.title = 'Remote sensing image written by REVERIE'
        net_ds.history = 'File created on ' + datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')
        net_ds.institution = 'AquaTel UQAR'
        net_ds.source = 'Remote sensing imagery'
        net_ds.version = '0.1.0'
        net_ds.references = 'https://github.com/raphidoc/reverie'
        net_ds.comment = 'Reflectance Extraction and Validation for Environmental Remote Imaging Exploration'

        # Create Dimensions
        net_ds.createDimension('W', len(self.wavelength))
        net_ds.createDimension('T', len([self.acq_time_z]))
        net_ds.createDimension('Z', len([self.z]))
        net_ds.createDimension('Y', len(self.y))
        net_ds.createDimension('X', len(self.x))

        # Create coordinate variables
        # We will store time as seconds since 1 january 1970 good luck people of 2038 :) !
        t_var = net_ds.createVariable('T', 'f4', ('T',))
        t_var.standard_name = 'time'
        t_var.long_name = 'UTC acquisition time of remote sensing image'
        t_var.units = 'seconds since 1970-01-01 00:00:00'
        t_var.calendar = 'gregorian'
        #epoch = cfunits.Units('seconds since 1970-01-01 00:00:00')
        t_var[:] = self.acq_time_z.timestamp()

        z_var = net_ds.createVariable('Z', 'f4', ('Z',))
        z_var.units = 'm'
        z_var.standard_name = 'altitude'
        z_var.long_name = 'Altitude is the viewing height above the geoid, positive upward'
        z_var.axis = 'y'
        z_var[:] = self.z

        y_var = net_ds.createVariable('Y', 'f4', ('Y',))
        y_var.units = 'm'
        y_var.standard_name = 'projection_y_coordinate'
        y_var.long_name = 'y-coordinate in projected coordinate system'
        y_var.axis = 'y'
        y_var[:] = self.y

        lat_var = net_ds.createVariable('lat', 'f4', ('Y',))
        lat_var.standard_name = 'latitude'
        lat_var.units = 'degrees_north'
        #lat_var.long_name = 'latitude'
        lat_var[:] = self.lat

        x_var = net_ds.createVariable('X', 'f4', ('X',))
        x_var.units = 'm'
        x_var.standard_name = 'projection_x_coordinate'
        x_var.long_name = 'x-coordinate in projected coordinate system'
        x_var.axis = 'x'
        x_var[:] = self.x

        lon_var = net_ds.createVariable('lon', 'f4', ('X',))
        lon_var.standard_name = 'longitude'
        lon_var.units = 'degrees_east'
        #lon_var.long_name = 'longitude'
        lon_var[:] = self.lon

        band_var = net_ds.createVariable('W', 'f4', ('W',))
        band_var.units = 'nm'
        band_var.standard_name = 'radiation_wavelength' # 'sensor_band_central_radiation_wavelength'
        band_var.long_name = 'Central wavelengths of the sensor bands'
        band_var.axis = 'Wavelength'
        band_var[:] = self.wavelength  # ([round(w, 2) for w in wavelengths])

        # grid_mapping
        crs = self.CRS
        print('Detected EPSG:' + str(crs.to_epsg()))
        cf_grid_mapping = crs.to_cf()

        proj_var = net_ds.createVariable('grid_mapping', np.int32, ())

        proj_var.grid_mapping_name = cf_grid_mapping['grid_mapping_name']
        proj_var.crs_wtk = cf_grid_mapping['crs_wkt']
        proj_var.semi_major_axis = cf_grid_mapping['semi_major_axis']
        proj_var.semi_minor_axis = cf_grid_mapping['semi_minor_axis']
        proj_var.inverse_flattening = cf_grid_mapping['inverse_flattening']
        proj_var.reference_ellipsoid_name = cf_grid_mapping['reference_ellipsoid_name']
        proj_var.longitude_of_prime_meridian = cf_grid_mapping['longitude_of_prime_meridian']
        proj_var.prime_meridian_name = cf_grid_mapping['prime_meridian_name']
        proj_var.geographic_crs_name = cf_grid_mapping['geographic_crs_name']
        proj_var.horizontal_datum_name = cf_grid_mapping['horizontal_datum_name']
        proj_var.projected_crs_name = cf_grid_mapping['projected_crs_name']
        proj_var.grid_mapping_name = cf_grid_mapping['grid_mapping_name']
        proj_var.latitude_of_projection_origin = cf_grid_mapping['latitude_of_projection_origin']
        proj_var.longitude_of_central_meridian = cf_grid_mapping['longitude_of_central_meridian']
        proj_var.false_easting = cf_grid_mapping['false_easting']
        proj_var.false_northing = cf_grid_mapping['false_northing']
        proj_var.scale_factor_at_central_meridian = cf_grid_mapping['scale_factor_at_central_meridian']

        self.proj_var = proj_var
        self.net_ds = net_ds

    def create_var_nc(self, var: str = None, dimensions: tuple = None, compression='zlib', complevel=1):
        """ Create a CF-1.0 variable in a NetCDF dataset

        Parameters
        ----------
        var : str
            Name of the variable to write to the NetCDF dataset
        dimensions : tuple
            tuple containing the dimension of the var with the form ('W', 'Y', 'X',).
        compression : str, default: 'zlib'
            Name of the compression algorithm to use. Use None to deactivate compression.
        complevel : int, default: 1
            Compression level.

        Returns
        -------

        See Also
        --------
        create_dataset_nc
        """

        ds = self.net_ds

        data_var = ds.createVariable(
            varname=var,
            datatype='i4',
            dimensions=dimensions,
            fill_value=self.no_data,
            compression=compression,
            complevel=complevel)  # ,
        # significant_digits=5)

        data_var.grid_mapping = self.proj_var.name

        # Follow the standard name table CF convention
        # TODO: fill the units and standard name automatically from var ?
        #  Water leaving reflectance could be: reflectance_of_water_leaving_radiance_on_incident_irradiance
        #  "reﬂectance based on the water-leaving radiance and the incident irradiance" (Ocean Optics Web Book)
        std_name, std_unit = get_cf_std_name(alias=var)

        data_var.units = std_unit
        data_var.standard_name = std_name
        #data_var.long_name = ''

        # self.__dst.variables['Rrs'].valid_min = 0
        # self.__dst.variables['Rrs'].valid_max = 6000
        data_var.missing_value = self.no_data

        '''
        scale_factor is used by NetCDF CF in writing and reading
        Reading: multiply by the scale_factor and add the add_offset
        Writing: subtract the add_offset and divide by the scale_factor
        If the scale factor is integer, to properly apply the scale_factor in the writing order we need the
        reciprocal of it.
        '''
        data_var.scale_factor = self.scale_factor
        data_var.add_offset = 0

    def to_nc(self):
        """
        TODO : Wrapper for create_dataset_nc, create_var_nc, write_var_nc, close_nc ?
        Returns
        -------
        """

    def from_nc(self, nc_file):
        """ Populate ReveCube object from NetCDF dataset

        Parameters
        ----------
        nc_file: str
            NetcCDF (.nc) CF-1.0 compliant file to read from

        Returns
        -------
        """

        if os.path.isfile(nc_file):
            net_ds = netCDF4.Dataset(nc_file, "r", format="NETCDF4")
            self.net_ds = net_ds
        else:
            raise Exception(f"File {nc_file} does not exist")

        # TODO: this not global but variable specific
        self.no_data = None
        self.scale_factor = None
        self.proj_var = None

        # Radiometric attributes
        wavelength_var = net_ds.variables['W']
        self.wavelength = wavelength_var[:].data

        # Geographic attributes
        altitude_var = net_ds.variables['Z']
        self.z = altitude_var[:][0] # As we have only one altitude, could be a scalar

        grid_mapping = net_ds.variables['grid_mapping']
        #pyproj.CRS.from_cf()
        self.CRS = pyproj.CRS.from_wkt(grid_mapping.crs_wtk)
        self.Affine = None
        self.n_rows = net_ds.dimensions['Y'].size
        self.n_cols = net_ds.dimensions['X'].size


        x_var = net_ds.variables['X']
        y_var = net_ds.variables['Y']
        lon_var = net_ds.variables['lon']
        lat_var = net_ds.variables['lat']
        self.x = x_var[:].data
        self.y = y_var[:].data
        self.lon = lon_var[:].data
        self.lat = lat_var[:].data
        self.lon_grid, self.lat_grid = None, None
        self.center_lon, self.center_lat = lon_var[round(len(self.x)/2)].data, lat_var[round(len(self.y)/2)].data

        # Time attributes
        time_var = net_ds.variables['T']

        self.acq_time_z = datetime.datetime.fromtimestamp(time_var[:][0])
        self.acq_time_local = None, None
        self.central_lon_local_timezone = None

        # Geometric attributes
        self.solar_zenith = None
        self.solar_azimuth = None

        self.viewing_zenith = None
        self.view_azimuth = None
        self.relative_azimuth = None
        self.sample_index = None