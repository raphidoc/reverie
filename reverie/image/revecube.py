# Standard library imports
import os
from abc import ABC
import datetime
import re

# Third party imports
import netCDF4
import numpy as np
import pyproj

# REVERIE import
from reverie.utils import helper
from reverie.utils.tile import Tile
from reverie.utils.cf_aliases import get_cf_std_name


class ReveCube(ABC):
    """
    ReveCube abstract class is used as the base template for spectral imagery data structure in REVERIE.
    This class is to be expanded by converter specific class as in ../converter/wise/read_pix.py

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
    sample_index: position of pixel on the detector array spatial dimension (pushbroom converter only)

        Methods
    -------
    cal_coordinate(): Compute the x, y projected coordinate from Affine, n_rows, n_cols and CRS
    """

    @classmethod
    def from_reve_nc(cls, nc_file):
        """Populate ReveCube object from reve CF NetCDF dataset

        Parameters
        ----------
        nc_file: str
            NetcCDF (.nc) CF-1.0 compliant file to read from

        Returns
        -------
        """

        if os.path.isfile(nc_file):
            src_ds = netCDF4.Dataset(nc_file, "r", format="NETCDF4")
        else:
            raise Exception(f"File {nc_file} does not exist")

        # TODO: better define the attributes to be read from the NetCDF file
        #  write all attribute that should be read from the NetCDF file

        # Radiometric attributes
        wavelength_var = src_ds.variables["W"]
        wavelength = wavelength_var[:].data

        # Geographic attributes
        altitude_var = src_ds.variables["Z"]
        z = altitude_var[:][0]  # As we have only one altitude, could be a scalar

        grid_mapping = src_ds.variables["grid_mapping"]
        crs = pyproj.CRS.from_wkt(grid_mapping.crs_wkt)
        affine = None
        n_rows = src_ds.dimensions["Y"].size
        n_cols = src_ds.dimensions["X"].size

        x_var = src_ds.variables["X"]
        y_var = src_ds.variables["Y"]
        lon_var = src_ds.variables["lon"]
        lat_var = src_ds.variables["lat"]
        x = x_var[:].data
        y = y_var[:].data
        lon = lon_var[:].data
        lat = lat_var[:].data
        lon_grid, lat_grid = None, None
        center_lon, center_lat = (
            lon_var[round(len(x) / 2)].data,
            lat_var[round(len(y) / 2)].data,
        )

        # Time attributes
        time_var = src_ds.variables["T"]

        acq_time_z = datetime.datetime.fromtimestamp(time_var[:][0])
        acq_time_local = None, None
        central_lon_local_timezone = None

        return cls(
            src_ds,
            wavelength,
            z,
            affine,
            n_rows,
            n_cols,
            crs,
            x,
            y,
            lon,
            lat,
            lon_grid,
            lat_grid,
            center_lon,
            center_lat,
            acq_time_z,
            acq_time_local,
            central_lon_local_timezone,
        )

    def __init__(
        self,
        src_ds: netCDF4.Dataset,
        wavelength: np.ndarray,
        z: float,
        affine,
        n_rows: int,
        n_cols: int,
        crs: pyproj.crs.CRS,
        x: np.ndarray,
        y: np.ndarray,
        lon: np.ndarray,
        lat: np.ndarray,
        lon_grid: np.ndarray,
        lat_grid: np.ndarray,
        center_lon: float,
        center_lat: float,
        acq_time_z: datetime.datetime,
        acq_time_local: datetime.datetime,
        central_lon_local_timezone: float,
    ):
        # Dataset attribute
        self.src_ds = src_ds
        self.nc_ds = None
        self.sensor = None
        self.no_data = None
        self._valid_mask = None
        # TODO: scale factor should be handled variable specific
        #  it might not be appropriate to apply the same to radiometric and geometric data for example
        #  Also not sure if it's good to give it a default value here
        self.scale_factor = 1
        self.proj_var = None

        # Radiometric attributes
        self.wavelength = wavelength

        # Geographic attributes
        # Altitude (z) could alternatively belong the Geometric attributes ?
        self.z = z
        self.Affine, self.n_rows, self.n_cols, self.CRS = affine, n_rows, n_cols, crs

        self.x, self.y, self.lon, self.lat = x, y, lon, lat
        self.lon_grid, self.lat_grid = lon_grid, lat_grid
        self.center_lon, self.center_lat = center_lon, center_lat

        # Time attributes
        self.acq_time_z, self.acq_time_local = acq_time_z, acq_time_local
        self.central_lon_local_timezone = central_lon_local_timezone

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
        return f"""
        Image from converter {self.sensor} acquired on {self.acq_time_z.strftime('%Y-%m-%d %H:%M:%SZ')}
        Central longitude: {self.center_lon:.3f}E
        Central latitude: {self.center_lat:.3f}N
        shape: x:{self.x.shape}, y:{self.y.shape}
        wavelength: {self.wavelength}
        """

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
        x, y = helper.transform_xy(
            affine, rows=list(range(n_rows)), cols=list(range(n_cols))
        )

        # transform the coordinates to WGS84 (EPSG:4326), extract longitude and latitude
        transformer = pyproj.Transformer.from_crs(crs.to_epsg(), 4326, always_xy=True)
        xv, yv = np.meshgrid(x, y)

        lon_grid, lat_grid = transformer.transform(xv, yv)

        lon, lat = lon_grid[0, :], lat_grid[:, 0]

        # print(f"lat shape: {lat.shape}, lon shape: {lon.shape}")
        print(
            f"n_rows({n_rows}) = y({len(y)}) = lat({len(lat)})\r\nn_cols({n_cols}) = x({len(x)}) = lon({len(lon)})"
        )

        # Could just use the lon lat vector
        central_lon = lon_grid[n_rows // 2, n_cols // 2]
        central_lat = lat_grid[n_rows // 2, n_cols // 2]

        self.lon_grid, self.lat_grid = lon_grid, lat_grid
        self.x, self.y, self.lon, self.lat, self.center_lon, self.center_lat = (
            x,
            y,
            lon,
            lat,
            central_lon,
            central_lat,
        )

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
        print(
            f"acquired UTC time:{self.acq_time_z}, and local time：{self.acq_time_local}"
        )

        offset_hours = self.acq_time_local.offset_hours
        self.central_lon_local_timezone = offset_hours * 15
        print(f"central longitude of {tz}:{self.central_lon_local_timezone}")

    def cal_sun_geom(self):
        """
        calculate solar zenith angle and azimuth angle
        :return:
        """
        hour_angle = helper.cal_solar_hour_angle(
            self.lon_grid, self.central_lon_local_timezone, self.acq_time_local
        )

        year, month, day = (
            self.acq_time_z.year,
            self.acq_time_z.month,
            self.acq_time_z.day,
        )

        declination = helper.cal_declination(year, month, day)
        self.solar_zenith, self.solar_azimuth = helper.calSolarZenithAzimuth(
            self.lat_grid, hour_angle, declination
        )

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
            return self._valid_mask[tile.sline : tile.eline, tile.spixl : tile.epixl]

    def cal_valid_mask(self):
        """
        Calculate the mask of valid pixel for the entire image (!= nodata)
        :return: _valid_mask
        """
        if self._valid_mask is None:
            iband = 0
            Lt = self.read_band(iband)
            self._valid_mask = Lt > 0

    def read_band(self, bandindex, tile: Tile = None):
        """
        read DN for a given band
        :param bandindex: bandindex starts with 0
        :param tile: an instance of the Tile class used for tiled processing
        :return: re
        """
        band_temp = self.src_ds.GetRasterBand(bandindex + 1)
        if tile:
            Lt = band_temp.ReadAsArray(
                xoff=tile[2], yoff=tile[0], win_xsize=tile.xsize, win_ysize=tile.ysize
            )
        else:
            Lt = band_temp.ReadAsArray()

        return Lt

    def cal_relative_azimuth(self):
        """
        calculate relative azimuth angle
        :return:
        """
        self.relative_azimuth = np.abs(self.view_azimuth - self.solar_azimuth)

    # @abstractmethod
    def cal_view_geom(self):
        pass

    def create_reve_nc(self, out_file: str = None):
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
            nc_ds = netCDF4.Dataset(out_file, "w", format="NETCDF4")
        except Exception as e:
            print(e)
            return

        # TODO validate that it follow the convention with cfdm / cf-python.
        #  For compatibility with GDAL NetCDF driver use CF-1.0
        nc_ds.Conventions = "CF-1.0"
        nc_ds.title = "Remote sensing image written by REVERIE"
        nc_ds.history = "File created on " + datetime.datetime.utcnow().strftime(
            "%Y-%m-%d %H:%M:%SZ"
        )
        nc_ds.institution = "AquaTel UQAR"
        nc_ds.source = "Remote sensing imagery"
        nc_ds.version = "0.1.0"
        nc_ds.references = "https://github.com/raphidoc/reverie"
        nc_ds.comment = "Reflectance Extraction and Validation for Environmental Remote Imaging Exploration"

        # Create Dimensions
        nc_ds.createDimension("W", len(self.wavelength))
        nc_ds.createDimension("T", len([self.acq_time_z]))
        nc_ds.createDimension("Z", len([self.z]))
        nc_ds.createDimension("Y", len(self.y))
        nc_ds.createDimension("X", len(self.x))

        band_var = nc_ds.createVariable("W", "f4", ("W",))
        band_var.units = "nm"
        band_var.standard_name = "radiation_wavelength"
        band_var.long_name = "Central wavelengths of the converter bands"
        band_var.axis = "Wavelength"
        band_var[:] = self.wavelength

        # Create coordinate variables
        # We will store time as seconds since 1 january 1970 good luck people of 2038 :) !
        t_var = nc_ds.createVariable("T", "f4", ("T",))
        t_var.standard_name = "time"
        t_var.long_name = "UTC acquisition time of remote sensing image"
        t_var.units = "seconds since 1970-01-01 00:00:00"
        t_var.calendar = "gregorian"
        t_var[:] = self.acq_time_z.timestamp()

        z_var = nc_ds.createVariable("Z", "f4", ("Z",))
        z_var.units = "m"
        z_var.standard_name = "altitude"
        z_var.long_name = (
            "Altitude is the viewing height above the geoid, positive upward"
        )
        z_var.axis = "y"
        z_var[:] = self.z

        y_var = nc_ds.createVariable("Y", "f4", ("Y",))
        y_var.units = "m"
        y_var.standard_name = "projection_y_coordinate"
        y_var.long_name = "y-coordinate in projected coordinate system"
        y_var.axis = "y"
        y_var[:] = self.y

        lat_var = nc_ds.createVariable("lat", "f4", ("Y",))
        lat_var.standard_name = "latitude"
        lat_var.units = "degrees_north"
        # lat_var.long_name = 'latitude'
        lat_var[:] = self.lat

        x_var = nc_ds.createVariable("X", "f4", ("X",))
        x_var.units = "m"
        x_var.standard_name = "projection_x_coordinate"
        x_var.long_name = "x-coordinate in projected coordinate system"
        x_var.axis = "x"
        x_var[:] = self.x

        lon_var = nc_ds.createVariable("lon", "f4", ("X",))
        lon_var.standard_name = "longitude"
        lon_var.units = "degrees_east"
        # lon_var.long_name = 'longitude'
        lon_var[:] = self.lon

        # grid_mapping
        crs = self.CRS
        print("Detected EPSG:" + str(crs.to_epsg()))
        cf_grid_mapping = crs.to_cf()

        grid_mapping = nc_ds.createVariable("grid_mapping", np.int32, ())

        grid_mapping.grid_mapping_name = cf_grid_mapping["grid_mapping_name"]
        grid_mapping.crs_wkt = cf_grid_mapping["crs_wkt"]
        grid_mapping.semi_major_axis = cf_grid_mapping["semi_major_axis"]
        grid_mapping.semi_minor_axis = cf_grid_mapping["semi_minor_axis"]
        grid_mapping.inverse_flattening = cf_grid_mapping["inverse_flattening"]
        grid_mapping.reference_ellipsoid_name = cf_grid_mapping[
            "reference_ellipsoid_name"
        ]
        grid_mapping.longitude_of_prime_meridian = cf_grid_mapping[
            "longitude_of_prime_meridian"
        ]
        grid_mapping.prime_meridian_name = cf_grid_mapping["prime_meridian_name"]
        grid_mapping.geographic_crs_name = cf_grid_mapping["geographic_crs_name"]
        grid_mapping.horizontal_datum_name = cf_grid_mapping["horizontal_datum_name"]
        grid_mapping.projected_crs_name = cf_grid_mapping["projected_crs_name"]
        grid_mapping.grid_mapping_name = cf_grid_mapping["grid_mapping_name"]
        grid_mapping.latitude_of_projection_origin = cf_grid_mapping[
            "latitude_of_projection_origin"
        ]
        grid_mapping.longitude_of_central_meridian = cf_grid_mapping[
            "longitude_of_central_meridian"
        ]
        grid_mapping.false_easting = cf_grid_mapping["false_easting"]
        grid_mapping.false_northing = cf_grid_mapping["false_northing"]
        grid_mapping.scale_factor_at_central_meridian = cf_grid_mapping[
            "scale_factor_at_central_meridian"
        ]

        self.proj_var = grid_mapping
        self.nc_ds = nc_ds

    def create_var_nc(
        self,
        var: str = None,
        datatype="i4",
        dimensions: tuple = None,
        scale_factor: float = None,
        compression="zlib",
        complevel=1,
    ):
        """Create a CF-1.0 variable in a NetCDF dataset

        Parameters
        ----------
        var : str
            Name of the variable to write to the NetCDF dataset
        dimensions : tuple
            Contain the dimension of the var with the form ('W', 'Y', 'X',).
        scale_factor : float
            scale_factor is used by NetCDF CF in writing and reading for lossy compression.
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

        ds = self.nc_ds

        data_var = ds.createVariable(
            varname=var,
            datatype=datatype,
            dimensions=dimensions,
            # fill_value=self.no_data,
            compression=compression,
            complevel=complevel,
        )

        data_var.grid_mapping = self.proj_var.name

        # Follow the standard name table CF convention
        # TODO: fill the units and standard name automatically from var ?
        #  Water leaving reflectance could be: reflectance_of_water_leaving_radiance_on_incident_irradiance
        #  "reﬂectance based on the water-leaving radiance and the incident irradiance" (Ocean Optics Web Book)
        std_name, std_unit = get_cf_std_name(alias=var)

        data_var.units = std_unit
        data_var.standard_name = std_name
        # data_var.long_name = ''

        # self.__dst.variables['Rrs'].valid_min = 0
        # self.__dst.variables['Rrs'].valid_max = 6000
        # Easier to leave missing_value as the default _FillValue
        # data_var.missing_value = self.no_data

        """
        scale_factor is used by NetCDF CF in writing and reading
        Reading: multiply by the scale_factor and add the add_offset
        Writing: subtract the add_offset and divide by the scale_factor
        If the scale factor is integer, to properly apply the scale_factor in the writing order we need the
        reciprocal of it.
        """
        data_var.scale_factor = scale_factor
        data_var.add_offset = 0

    # def to_nc(self):
    #     """
    #     TODO : Wrapper for create_dataset_nc, create_var_nc, write_var_nc, close_nc ?
    #     Returns
    #     -------
    #     """
