# Standard library imports
import logging
import os
from abc import ABC
import datetime
import re
import math

# Third party imports
import netCDF4
import numpy as np
import pyproj
import pandas as pd
import geopandas as gpd
from statsmodels.tsa.forecasting.theta import extend_index
from tqdm import tqdm
from p_tqdm import p_uimap
import shapely
import xarray as xr
import zarr
import affine
from rasterio.windows import Window
from shapely.geometry import Point, box

# REVERIE import
from reverie.utils import helper, astronomy
from reverie.image.tile import Tile
from reverie.utils.cf_aliases import get_cf_std_name


class ReveCube(ABC):
    """
    ReveCube class is used as the base template for spectral imagery data structure in REVERIE.
    This class is to be expanded by converter specific class as in ../converter/wise/read_pix.py

    The class is designed to be used with the xarray library to handle multi-dimensional arrays.

    Parameters
    ----------
    in_path: str
        Path to the input dataset
    in_ds: xarray.Dataset
        xarray Dataset object
    wavelength: np.ndarray
        Array of wavelength
    acq_time_z: datetime
        Image acquisition time in UTC
    z: float
        Altitude of the sensor
    y: np.ndarray
        Array of y coordinates
    x: np.ndarray
        Array of x coordinates
    lat: np.ndarray
        Array of latitude
    lon: np.ndarray
        Array of longitude
    n_rows: int
        Number of rows
    n_cols: int
        Number of columns
    Affine: Affine
        Affine transformation
    crs: pyproj.crs.CRS
        Coordinate Reference System

    Attributes
    ----------

    Methods
    -------
    from_zarr
    from_reve_nc
    decode_xr
    __init__
    __str__
    cal_coordinate
    cal_coordinate_grid
    cal_time
    cal_sun_geom
    get_valid_mask
    cal_valid_mask

    """

    @classmethod
    def from_zarr(cls, in_path: str):
        """Populate ReveCube object from zarr store

        Parameters
        ----------
        src_store: str
            zarr store to read from

        Returns
        -------
        """

        if os.path.isdir(in_path):
            in_ds = xr.open_zarr(in_path, consolidated=True, decode_times=False)
        else:
            raise Exception(f"Directory {in_path} does not exist")

        return cls.decode_xr(in_path, in_ds)

    @classmethod
    def from_reve_nc(cls, in_path: str):
        """Populate ReveCube object from reve CF NetCDF dataset

        Parameters
        ----------
        in_path: str
            NetcCDF (.nc) CF-1.0 compliant file to read from

        Returns
        -------
        """

        if os.path.isfile(in_path):
            in_ds = xr.open_dataset(in_path, decode_times=False)

            # in_ds = netCDF4.Dataset(in_path, "r+", format="NETCDF4")
            # image_name = os.path.basename(in_path).split(".")[0]
            # in_ds.image_name = image_name
            # in_ds.close()
        else:
            raise Exception(f"File {in_path} does not exist")

        return cls.decode_xr(in_path, in_ds)

    @classmethod
    def decode_xr(cls, in_path, in_ds: xr.Dataset):
        # Image name
        image_name = in_ds.attrs["image_name"]

        # Spectral attributes
        wavelength = in_ds.variables["wavelength"].data

        # Spatiotemporal attributes
        time_var = in_ds.variables["time"]
        # if time is decoded by xarray as datetime64[ns], convert it to datetime.datetime
        # ts = (
        #     time_var.data[0] - np.datetime64("1970-01-01T00:00:00Z")
        # ) / np.timedelta64(1, "s")
        # TODO: better define acq_time_z as image center datetime,
        #  also should it be decoded by xarray or not ?
        # For some reason utcfromtimestamp doesn't carry tz info (acutally UTC is not a timezone, GMT is the timezone)
        # It create a problem if the data loaded by panda follow the iso8601 format.
        # In that case the datetime object will have a utc timezone.
        # We cannot make comutation on timezone aware and not aware.
        # It's best to force the timezone to be utc and to let the user actually deal with the data provided to the app.
        acq_time_z = datetime.datetime.utcfromtimestamp(int(time_var.data[0])).replace(
            tzinfo=datetime.timezone.utc
        )
        altitude_var = in_ds.variables["z"]
        z = altitude_var[:][0]
        y = in_ds.variables["y"][:].data
        x = in_ds.variables["x"][:].data
        n_rows = in_ds.sizes["y"]
        n_cols = in_ds.sizes["x"]

        grid_mapping = in_ds.variables["grid_mapping"]

        # Get affine_transform and crs from grid_mapping
        a, b, c, d, e, f = grid_mapping.attrs["affine_transform"]
        Affine = affine.Affine(a, b, c, d, e, f)
        crs = pyproj.CRS.from_wkt(grid_mapping.attrs["crs_wkt"])

        return cls(
            in_path,
            in_ds,
            image_name,
            wavelength,
            acq_time_z,
            z,
            y,
            x,
            n_rows,
            n_cols,
            Affine,
            crs,
        )

    def __init__(
        self,
        in_path: str,
        in_ds: xr.Dataset,
        image_name: str,
        wavelength: np.ndarray,
        acq_time_z: datetime.datetime,
        z: float,
        y: np.ndarray,
        x: np.ndarray,
        n_rows: int,
        n_cols: int,
        Affine,
        crs: pyproj.crs.CRS,
    ):
        # Dataset accessor
        self.in_path = in_path
        self.in_ds = in_ds
        self.out_path = None
        self.out_ds = None

        self.image_name = image_name

        # Spectral attributes
        self.wavelength = wavelength

        # Spatiotemporal attributes
        self.acq_time_z = acq_time_z
        self.z = z
        self.y = y
        self.x = x
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.Affine = Affine
        self.CRS = crs

        self.center_lon = None
        self.center_lat = None
        self.lon_grid = None
        self.lat_grid = None

        # Sensor attribute store metadata from the sensor in a dictionary
        self.sensor = {}

        # Optional attributes

        self.no_data = None
        self.valid_mask = None
        self.bad_band_list = None

        self.water_mask = None

        # Time attributes
        self.acq_time_local = None
        self.central_lon_local_timezone = None

        # Geometric attributes
        self.sun_zenith = None
        self.sun_azimuth = None
        self.view_zenith = None
        self.view_azimuth = None
        self.relative_azimuth = None

        # Pixel location on the sensor array
        self.sample_index = None
        # Scan line location on the image
        self.line_index = None

    def __str__(self):
        return f"""
        Image from sensor {self.sensor} acquired on {self.acq_time_z.strftime('%Y-%m-%d %H:%M:%SZ')}
        Central longitude: {self.center_lon:.3f}E
        Central latitude: {self.center_lat:.3f}N
        shape: x:{self.x.shape}, y:{self.y.shape}
        wavelength: {self.wavelength}
        """

    # def cal_coordinate(self, affine, n_rows, n_cols, crs):
    #     """
    #     Compute the pixel coordinates
    #
    #     Parameters
    #     ----------
    #     affine: Affine transformation
    #     n_rows: Number of rows
    #     n_cols: Number of column
    #     crs: pyproj CRS object
    #
    #     Returns
    #     -------
    #     x: projected pixel coordinates
    #     y: projected pixel coordinates
    #     longitude:
    #     latitude:
    #     center_longitude:
    #     center_latitude:
    #     """
    #     logging.debug("calculating pixel coordinates")
    #
    #     # Define image projected coordinates from Affine tranformation
    #     # TODO: it appear that x, y doesn't match lon lat ....
    #     x, y = helper.transform_xy(
    #         affine, rows=list(range(n_rows)), cols=list(range(n_cols))
    #     )
    #
    #     # compute longitude and latitude from the projected coordinates
    #     transformer = pyproj.Transformer.from_crs(crs.to_epsg(), 4326, always_xy=True)
    #
    #     xv, yv = np.meshgrid(x, y[0])
    #     lon, _ = transformer.transform(xv, yv)
    #     lon = lon[0, :]
    #
    #     xv, yv = np.meshgrid(x[0], y)
    #     _, lat = transformer.transform(xv, yv)
    #     lat = lat[:, 0]
    #
    #     # print(f"lat shape: {lat.shape}, lon shape: {lon.shape}")
    #     logging.info(
    #         f"n_rows({n_rows}) = y({len(y)}) = lat({len(lat)}) and n_cols({n_cols}) = x({len(x)}) = lon({len(lon)})"
    #     )
    #
    #     self.x, self.y, self.lon, self.lat = (x, y, lon, lat)
    #
    def cal_coordinate(self, affine, n_rows, n_cols):
        logging.debug("calculating pixel coordinates")

        # Projected coordinates
        x, y = helper.transform_xy(
            affine, rows=list(range(n_rows)), cols=list(range(n_cols))
        )

        logging.info(
            f"n_rows({n_rows}) = y({len(y)}) and "
            f"n_cols({n_cols}) = x({len(x)})"
        )

        self.x, self.y = x, y

    def expand_coordinate(self):
        # Transformer to WGS84
        transformer = pyproj.Transformer.from_crs(self.CRS.to_epsg(), 4326, always_xy=True)

        # 2D meshgrid for all pixel positions
        xv, yv = np.meshgrid(self.x, self.y)
        lon_grid, lat_grid = transformer.transform(xv, yv)

        central_lon = lon_grid[len(lat_grid[:, 0]) // 2, len(lon_grid[0, :]) // 2]
        central_lat = lat_grid[len(lat_grid[:, 0]) // 2, len(lon_grid[0, :]) // 2]

        self.lon_grid, self.lat_grid = lon_grid, lat_grid
        self.center_lon, self.center_lat = central_lon, central_lat

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
        logging.info(
            f"acquired UTC time:{self.acq_time_z}, and local time：{self.acq_time_local}"
        )

        offset_hours = self.acq_time_local.utcoffset().total_seconds() / 3600

        self.central_lon_local_timezone = offset_hours * 15
        logging.info(f"central longitude of {tz}:{self.central_lon_local_timezone}")

    def cal_sun_geom(self):
        """calculate solar zenith and azimuth angle for each pixel in the scene
        Parameters
        ----------
        acq_time_local: local time of image acquisition
        lat_grid: latitude grid
        lon_grid: longitude grid

        Returns
        -------
        sun_zenith: spatially resolved solar zenith [degree]
        sun_azimuth: spatially resolved solar azimuth [degree]
        """

        logging.debug("calculating solar zenith and azimuth")

        utc_offset = self.acq_time_local.utcoffset().total_seconds() / 3600

        self.sun_zenith, self.sun_azimuth = astronomy.sun_geom_noaa(
            self.acq_time_local, utc_offset, self.lat_grid, self.lon_grid
        )

        self.sun_zenith[~self.get_valid_mask()] = np.nan
        self.sun_azimuth[~self.get_valid_mask()] = np.nan

    def get_valid_mask(self, window: Window = None):
        """
        Get the valid mask for the enire image or a specific tile
        :param window: an object of class Window
        :return:[]
        """
        # logging.debug("geting valid mask")
        if self.valid_mask is None:
            self.cal_valid_mask()
        if window is None:
            return self.valid_mask
        else:
            y_start = int(window.row_off)
            y_stop = y_start + int(window.height)
            x_start = int(window.col_off)
            x_stop = x_start + int(window.width)
            #return self.valid_mask[tile.sline : tile.eline, tile.spixl : tile.epixl]
            return self.valid_mask[y_start : y_stop, x_start : x_stop]

    def cal_valid_mask(self):
        """
        Calculate the mask of valid pixel for the entire image (!= nodata)
        :return: valid_mask
        """
        logging.debug("calculating valid mask")
        if self.valid_mask is None:
            # Select the first variable with dim (wavelength, y, x) to compute the valid mask
            for name, data in self.in_ds.data_vars.items():
                if data.dims == ("wavelength", "y", "x"):
                    # TODO: some bands might be full of NA, refer to attribute bad_band_list
                    self.valid_mask = data.isel(wavelength=60).notnull().data
                    break

    def filter_bad_band(self):
        # filter image for bad bands
        bad_band_list = self.in_ds["radiance_at_sensor"].bad_band_list
        if isinstance(bad_band_list, str):
            bad_band_list = str.split(bad_band_list, ", ")

        bad_band_list = np.array(bad_band_list)
        good_band_indices = np.where(bad_band_list == "1")[0]
        good_bands_slice = slice(min(good_band_indices), max(good_band_indices) + 1)

        self.wavelength = self.wavelength[good_band_indices]
        self.in_ds = self.in_ds.isel(wavelength=good_bands_slice)

    def mask_wavelength(self, wavelength):
        wavelength_mask = (self.wavelength >= min(wavelength)) & (self.wavelength <= max(wavelength))

        self.wavelength = self.wavelength[wavelength_mask]
        self.in_ds = self.in_ds.sel(wavelength=self.wavelength)


    def read_band(self, bandindex, tile: Tile = None):
        """
        read DN for a given band
        :param bandindex: bandindex starts with 0
        :param tile: an instance of the Tile class used for tiled processing
        :return: re
        """
        pass

    def cal_relative_azimuth(self):
        """
        calculate relative azimuth angle
        :return:
        The angle between the viewing and the sun azimuth retricted to 0 - 180 degree, 0 facing towards the sun and 180 facing away from the sun
        """
        # In ACOLTIE and 6S relative azimuth is defined as theta_s - theta_v in the range 0 - 180
        # if 0  then theta_s = theta_v, viewing is in the incidence direction of the sun ray, looking away from it
        # if 180 then viewing is in the opposite direction of the sun's ray, looking toward it
        relative_azimuth = np.abs(self.view_azimuth - self.sun_azimuth)

        if 0 <= relative_azimuth.all() <= 180:
            self.relative_azimuth = relative_azimuth
        elif 180 < relative_azimuth.all() <= 360:
            self.relative_azimuth = 360 - relative_azimuth
        else:
            raise ValueError("relative azimuth is not in a valid 0 - 360 range")

    # @abstractmethod
    def cal_view_geom(self):
        pass

    def update_attributes(self):
        self.wavelength = self.in_ds.wavelength.values
        self.y = self.in_ds.y
        self.x = self.in_ds.x
        self.n_rows = len(self.y)
        self.n_cols = len(self.x)
        # New affine origin
        a, b, c, d, e, f, g, h, i = self.Affine
        # New origin (top-left corner)
        c = self.x[0].values
        f = self.y[0].values
        self.Affine = affine.Affine(a, b, c, d, e, f)

    def create_windows(self, window_size: int) -> list[Window]:
        n_y = len(self.y)
        n_x = len(self.x)
        windows = []
        for y_start in range(0, n_y, window_size):
            y_end = min(y_start + window_size, n_y)
            height = y_end - y_start
            for x_start in range(0, n_x, window_size):
                x_end = min(x_start + window_size, n_x)
                width = x_end - x_start
                windows.append(Window(x_start, y_start, width, height))
        return windows

    def create_reve_nc(self, out_path: str = None):
        """Create CF-1.0 compliant NetCDF dataset from DateCube
        CF-1.0 (http://cfconventions.org/Data/cf-conventions/cf-conventions-1.0/build/cf-conventions.html#dimensions)
         is chosen to ensure compatibility with the GDAL NetCDF driver:
         (https://gdal.org/drivers/raster/netcdf.html).

        This format can also be read by SNAP but the spectrum view tool does not find any spectral dimension.
        Quinten wrote something about that in ACOLITE.

        :return: None
        """

        if out_path is None:
            raise Exception("out_path file not set, cannot create dataset")

        try:
            nc_ds = netCDF4.Dataset(out_path, "w", format="NETCDF4")
        except Exception as e:
            print(e)
            return

        self.out_path = out_path

        # TODO validate that it follow the convention with cfdm / cf-python.
        #  For compatibility with GDAL NetCDF driver use CF-1.0
        #  https://cfconventions.org/conventions.html
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
        nc_ds.image_name = os.path.basename(self.in_path).split(".")[0]

        # Create Dimensions
        nc_ds.createDimension("wavelength", len(self.wavelength))
        nc_ds.createDimension("time", len([self.acq_time_z]))
        nc_ds.createDimension("z", len([self.z]))
        nc_ds.createDimension("y", len(self.y))
        nc_ds.createDimension("x", len(self.x))

        band_var = nc_ds.createVariable("wavelength", "f4", ("wavelength",))
        band_var.units = "nm"
        band_var.standard_name = "radiation_wavelength"
        band_var.long_name = "Central wavelengths of the converter bands"
        band_var.axis = "wavelength"
        band_var[:] = self.wavelength

        # Create coordinate variables
        # We will store time as seconds since 1 january 1970 good luck people of 2038 :) !
        t_var = nc_ds.createVariable("time", "f8", ("time",))
        t_var.standard_name = "time"
        t_var.long_name = "UTC acquisition time of remote sensing image"
        # CF convention for time zone is UTC if ommited
        # xarray will convert it to a datetime64[ns] object considering it is local time
        t_var.units = "seconds since 1970-01-01 00:00:00 +00:00"
        t_var.calendar = "gregorian"
        t_var[:] = self.acq_time_z.timestamp()

        z_var = nc_ds.createVariable("z", "f8", ("z",))
        z_var.units = "m"
        z_var.standard_name = "altitude"
        z_var.long_name = (
            "Altitude is the viewing height above the geoid, positive upward"
        )
        z_var.axis = "z"
        z_var[:] = self.z

        y_var = nc_ds.createVariable("y", "f8", ("y",))
        y_var.units = "m"
        y_var.standard_name = "projection_y_coordinate"
        y_var.long_name = "y-coordinate in projected coordinate system"
        y_var.axis = "y"
        y_var[:] = self.y

        # lat_var = nc_ds.createVariable("lat", "f8", ("y",))
        # lat_var.standard_name = "latitude"
        # lat_var.units = "degrees_north"
        # # lat_var.long_name = 'latitude'
        # lat_var[:] = self.lat

        x_var = nc_ds.createVariable("x", "f8", ("x",))
        x_var.units = "m"
        x_var.standard_name = "projection_x_coordinate"
        x_var.long_name = "x-coordinate in projected coordinate system"
        x_var.axis = "x"
        x_var[:] = self.x

        # lon_var = nc_ds.createVariable("lon", "f8", ("x",))
        # lon_var.standard_name = "longitude"
        # lon_var.units = "degrees_east"
        # # lon_var.long_name = 'longitude'
        # lon_var[:] = self.lon

        # grid_mapping
        crs = self.CRS
        logging.info("Detected EPSG:" + str(crs.to_epsg()))
        cf_grid_mapping = crs.to_cf()

        grid_mapping = nc_ds.createVariable("grid_mapping", np.int32, ())

        # Fix GDAL projection issue by providing the attributes spatial_ref and GeoTranform
        # GeoTransform is an affine transform array defined by GDAL with parameters in the orders:
        # GT0 = 541107.0  # x coordinate of the upper-left corner of the upper-left pixel
        # GT1 = 1.5  # w-e pixel resolution / pixel width.
        # GT2 = 0  # row rotation (typically zero).
        # GT3 = (
        #     5438536.5  # y-coordinate of the upper-left corner of the upper-left pixel.
        # )
        # GT4 = 0  # column rotation (typically zero).
        # GT5 = 1.5  # n-s pixel resolution / pixel height (negative value for a north-up image).
        # grid_mapping.GeoTransform = f"{GT0} {GT1} {GT2} {GT3} {GT4} {GT5}"
        # raise "Implement this first !"

        grid_mapping.spatial_ref = cf_grid_mapping["crs_wkt"]

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

        grid_mapping.affine_transform = (
            self.Affine.a,
            self.Affine.b,
            self.Affine.c,
            self.Affine.d,
            self.Affine.e,
            self.Affine.f,
        )

        # self.proj_var = grid_mapping
        self.out_ds = nc_ds

    def create_var_nc(
        self,
        name: str = None,
        type=None,
        dims: tuple = None,
        scale: float = 1.0,
        comp="zlib",
        complevel=1,
    ):
        """Create a CF-1.0 variable in a NetCDF dataset

        Parameters
        ----------
        name : str
            Name of the variable to write to the NetCDF dataset
        type : str
            Data type of the variable to write to the NetCDF dataset
        dims : tuple
            Contain the dimension of the var with the form ('wavelength', 'y', 'x',).
        scale : float
            scale_factor is used by NetCDF CF in writing and reading for lossy compression.
        comp : str, default: 'zlib'
            Name of the compression algorithm to use. Use None to deactivate compression.
        complevel : int, default: 1
            Compression level.

        Returns
        -------

        See Also
        --------
        create_dataset_nc
        """

        ds = self.out_ds

        # Easier to leave missing_value as the default _FillValue,
        # but then GDAL doesn't recognize it ...
        # self.no_data = netCDF4.default_fillvals[type]
        # When scaling the default _FillValue, it get somehow messed up when reading with GDAL

        # Offset no_data to fit inside data type after scaling
        # self.no_data = math.trunc(netCDF4.default_fillvals[type] * 0.00001)
        self.no_data = math.trunc(netCDF4.default_fillvals[type] * scale)

        # if scale == 1.0:
        #     self.no_data = netCDF4.default_fillvals[type] - 50

        data_var = ds.createVariable(
            varname=name,
            datatype=type,
            dimensions=dims,
            fill_value=self.no_data,
            compression=comp,
            complevel=complevel,
        )

        data_var.grid_mapping = "grid_mapping"  # self.proj_var.name

        # Follow the standard name table CF convention
        # TODO: fill the units and standard name automatically from name ?
        #  Water leaving reflectance could be: reflectance_of_water_leaving_radiance_on_incident_irradiance
        #  "reﬂectance based on the water-leaving radiance and the incident irradiance" (Ocean Optics Web Book)
        std_name, std_unit = get_cf_std_name(alias=name)

        data_var.units = std_unit
        data_var.standard_name = std_name
        # data_var.long_name = ''

        # self.__dst.variables['rho_remote_sensing'].valid_min = 0
        # self.__dst.variables['rho_remote_sensing'].valid_max = 6000
        data_var.missing_value = self.no_data

        """
        scale is used by NetCDF CF in writing and reading
        Reading: multiply by the scale and add the add_offset
        Writing: subtract the add_offset and divide by the scale
        If the scale factor is integer, to properly apply the scale in the writing order we need the
        reciprocal of it.
        """
        data_var.scale_factor = scale
        data_var.add_offset = 0

    def to_reve_nc(self):
        """
        TODO : Wrapper for create_dataset_nc, create_var_nc, write_var_nc, close_nc ?
        Returns
        -------
        """
        pass

    def to_zarr(self, out_store: str = None):
        if out_store is None:
            raise Exception("out_path file not set, cannot create dataset")

        try:
            store = zarr.DirectoryStore("data/example.zarr")
        except Exception as e:
            print(e)
            return

        # TODO: write a reve cube to a zarr store
        pass

    def get_footprint(self):

        import rasterio.features
        from shapely.geometry import shape

        # Filter observation outside the image valid_mask
        valid_mask = self.get_valid_mask()

        # Convert the valid_mask to list of polygon(s)
        polygons = [
            shape(geom)
            for geom, value in rasterio.features.shapes(
                valid_mask.astype("int16"), mask=valid_mask, transform=self.Affine
            )
            if value == 1
        ]

        # Convert the list of polygons to a GeoDataFrame
        footprint_gdf = gpd.GeoDataFrame(geometry=polygons, crs=self.CRS.to_epsg())

        return footprint_gdf


    def extract_pixel(
        self,
        matchup_file: str = None,
        var_name: list = None,
        max_time_diff: float = None,
        window_size: int = 1,
        output_box: str = None
    ):
        """
        Extract pixel value from a list of in-situ data

        Parameters
        ----------
        matchup_file : str
            Path to the in-situ data file
        var_name : list
            List of the variable name of image data to extract
        max_time_diff : float
            Maximum time difference between the in-situ data and the image acquisition time, in hours.
        window_size : int
            Size of the window used to extract pixel value from the image.
            Unit is pixel and must be an odd number (i.e 1,3,5,7,...).

        Returns
        -------
        pixex_df : pandas.DataFrame
            A dataframe containing the extracted pixel value
        """

        # def extractor(uuid, xr_ds, matchup_gdf=matchup_gdf):

        # iterator = p_uimap(extractor, matchup_gdf["UUID"])

        # load the nc dataset with xarray
        xr_ds = self.in_ds

        matchup_gdf = pd.read_csv(matchup_file)
        matchup_gdf.columns = matchup_gdf.columns.str.lower()
        matchup_gdf = matchup_gdf[["date_time", "lat", "lon", "uuid"]]

        # When matchup data is in long format (wavelength along the rows), keep unique observation
        matchup_gdf = matchup_gdf.drop_duplicates()

        # Filter observation outside requested time range
        if max_time_diff:
            matchup_gdf["date_time"] = pd.to_datetime(matchup_gdf["date_time"], utc=True)
            matchup_gdf = matchup_gdf[
                abs(matchup_gdf["date_time"] - self.acq_time_z)
                < pd.Timedelta(max_time_diff, unit="h")
            ]
            logging.info(
                "%s observations remaining after time filtering" % len(matchup_gdf)
            )

            if len(matchup_gdf) == 0:
                raise Exception("no matchup data remaining after time filter")

        # Project matchup data to image
        match_geometry = gpd.points_from_xy(
            matchup_gdf["lon"], matchup_gdf["lat"], crs="EPSG:4326"
        )

        matchup_gdf = gpd.GeoDataFrame(matchup_gdf, geometry=match_geometry)

        logging.info(f"Projecting in-situ data to EPSG: {self.CRS.to_epsg()}")
        matchup_gdf = matchup_gdf.to_crs(self.CRS.to_epsg())

        # import rasterio.features
        # from shapely.geometry import shape
        #
        # # Filter observation outside the image valid_mask
        # valid_mask = self.get_valid_mask()
        #
        # # Convert the valid_mask to list of polygon(s)
        # polygons = [
        #     shape(geom)
        #     for geom, value in rasterio.features.shapes(
        #         valid_mask.astype("int16"), mask=valid_mask, transform=self.Affine
        #     )
        #     if value == 1
        # ]
        #
        # # Convert the list of polygons to a GeoDataFrame
        # footprint_gdf = gpd.GeoDataFrame(geometry=polygons, crs=matchup_gdf.crs)
        #
        # # Write the GeoDataFrame to a GeoJSON file
        # # footprint_gdf.to_file("footprint.geojson", driver="GeoJSON")

        # Perform a spatial join between the matchup_gdf and the polygons_gdf
        footprint_gdf = self.get_footprint()

        filtered_gdf = gpd.sjoin(
            matchup_gdf, footprint_gdf, how="inner", predicate="intersects"
        )

        logging.info("%s observation remaining after spatial filtering" % len(filtered_gdf))

        if len(filtered_gdf) == 0:
            raise Exception("no matchup data remaining after spatial filter")

        pixex_df = pd.DataFrame()
        window_dict = {}
        window_data = {}

        for uuid in tqdm(filtered_gdf["uuid"]):
            temp_gdf = filtered_gdf[filtered_gdf["uuid"] == uuid].reset_index()

            # Make sure coordinates are in raster CRS
            x_coords, y_coords = xr_ds.x.values, xr_ds.y.values
            x_pt = temp_gdf.geometry.x.iloc[0]
            y_pt = temp_gdf.geometry.y.iloc[0]

            x_index = np.abs(x_coords - x_pt).argmin()
            y_index = np.abs(y_coords - y_pt).argmin()

            half_window = window_size // 2
            data_sel = xr_ds.isel(
                y=slice(max(0, y_index - half_window), y_index + half_window + 1),
                x=slice(max(0, x_index - half_window), x_index + half_window + 1),
            )

            # print("UTM20N x = " + str(data_sel.x[1].item()))
            # print("UTM20N y = " + str(data_sel.y[1].item()))
            #
            # # Get UTM20N x, y values
            # x = data_sel.x[1].item()
            # y = data_sel.y[1].item()
            #
            # # Define the transformer from UTM20N (EPSG:32620) to WGS84 (EPSG:4326)
            # transformer = pyproj.Transformer.from_crs("EPSG:32620", "EPSG:4326", always_xy=True)
            #
            # lon, lat = transformer.transform(x, y)
            #
            # print("lat = " + str(lat))
            # print("lon = " + str(lon))

            # Pick the exact pixel center in lat/lon
            # lat = xr_ds.lat.isel(y=y_index).item()
            # lon = xr_ds.lon.isel(x=x_index).item()

            # print("lat = " + str(lat))
            # print("lon = " + str(lon))
            #
            # import pyproj
            #
            # # Define the transformer from WGS84 (EPSG:4326) to UTM20N (EPSG:32620)
            # transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32620", always_xy=True)
            #
            # x, y = transformer.transform(lon, lat)
            #
            # print("UTM20N x = " + str(x))
            # print("UTM20N y = " + str(y))

            # Point geometry (same CRS as raster)
            # point_geom = gpd.GeoDataFrame(
            #     {'uuid': [uuid]},
            #     geometry=[Point(data_sel.x[half_window], data_sel.y[half_window])],
            #     crs=self.CRS.to_epsg()
            # )
            #
            # point_geom.to_file(os.path.join(output_box, f"{uuid}_point.geojson"), driver="GeoJSON")

            # Window bounds
            win = Window.from_slices(
                rows=slice(max(0, y_index - half_window), y_index + half_window + 1),
                cols=slice(max(0, x_index - half_window), x_index + half_window + 1),
            )
            window_dict[uuid] = win
            window_data[uuid] = data_sel

            # y_start, y_stop = int(win.row_off), int(win.row_off + win.height)
            # x_start, x_stop = int(win.col_off), int(win.col_off + win.width)
            #
            # x0, y0 = helper.transform_xy(self.Affine, rows=[y_start - 0.5], cols=[x_start - 0.5])
            # x1, y1 = helper.transform_xy(self.Affine, rows=[y_stop - 0.5], cols=[x_stop - 0.5])
            #
            # box_geom = gpd.GeoDataFrame(
            #     {'uuid': [uuid]},
            #     geometry=[box(x0[0], y0[0], x1[0], y1[0])],
            #     crs=self.CRS.to_epsg()
            # )
            # box_geom.to_file(os.path.join(output_box, f"{self.image_name}_windows.geojson"), driver="GeoJSON")

            # import holoviews as hv
            # from holoviews import opts
            # import panel as pn
            #
            # # setting bokeh as backend
            # hv.extension("bokeh")
            #
            # opts.defaults(
            #     opts.GridSpace(shared_xaxis=True, shared_yaxis=True),
            #     opts.Image(cmap="viridis", width=400, height=400),
            #     opts.Labels(
            #         text_color="white",
            #         text_font_size="8pt",
            #         text_align="left",
            #         text_baseline="bottom",
            #     ),
            #     opts.Path(color="white"),
            #     opts.Spread(width=600),
            #     opts.Overlay(show_legend=False),
            # )
            #
            # ds = hv.Dataset(data_sel, vdims=["radiance_at_sensor"])
            #
            # plot = ds.to(hv.Image, kdims=["x", "y"], dynamic=True).hist()
            #
            # renderer = hv.renderer("bokeh")
            # renderer.save(
            #     plot, "/D/Documents/PhD/Thesis/Chapter2/Data/WISE/pixex/test.html"
            # )
            #
            # pn.serve(plot)

            temp_pixex_df = data_sel.to_dataframe()
            # temp_pixex_df = temp_pixex_df.rename_axis("Wavelength")
            # temp_pixex_df = temp_pixex_df.reset_index()
            # TODO output a wide format when wavelength and non wavelength data are mixed ?
            # temp_pixex_df = pd.pivot(temp_pixex_df, index=['x', 'y'], columns='Wavelength', values='radiance_at_sensor')
            temp_pixex_df["uuid"] = uuid
            temp_pixex_df["image_name"] = self.image_name
            # temp_pixex_df['Sensor'] = str(temp_pix_ex_array.Sensor.values)
            # temp_pixex_df['ImageDate'] = str(temp_pix_ex_array.coords['isodate'].values)
            # temp_pixex_df['AtCor'] = 'ac4icw'

            pixex_df = pd.concat([pixex_df, temp_pixex_df], axis=0)

        if output_box is not None and os.path.isdir(output_box):

            polygons = []
            uuids = []

            for uuid, window in window_dict.items():
                # Get pixel coordinates from window slices
                y_start = int(window.row_off)
                y_stop = y_start + int(window.height)
                x_start = int(window.col_off)
                x_stop = x_start + int(window.width)

                # Get projected coordinates using affine transform
                x0, y0 = helper.transform_xy(self.Affine, rows=[y_start - 0.5], cols=[x_start - 0.5])
                x1, y1 = helper.transform_xy(self.Affine, rows=[y_stop - 0.5], cols=[x_stop - 0.5])

                # Create polygon (bounding box)
                poly = box(x0[0], y0[0], x1[0], y1[0])
                polygons.append(poly)
                uuids.append(uuid)

            gdf = gpd.GeoDataFrame({'uuid': uuids, 'geometry': polygons}, crs=self.CRS.to_epsg())
            geojson_path = os.path.join(output_box, self.image_name + "_windows.geojson")
            gdf.to_file(geojson_path, driver="GeoJSON")

        return pixex_df, window_dict, window_data

    def extract_pixel_line(self, line_index: int, line_window: int):
        xr_ds = self.in_ds

        # Stack the 'y' and 'x' dims into a single dimension 'yx'
        stacked_ds = xr_ds.stack(yx=("y", "x"))

        # Assign 'LineIndex' as a coordinate to the stacked Dataset
        LineIndex = xr_ds["LineIndex"].values.flatten()
        stacked_ds = stacked_ds.assign_coords(LineIndex=("yx", LineIndex))

        # Swap the 'yx' dimension with 'LineIndex'
        xr_ds_swapped = stacked_ds.swap_dims({"yx": "LineIndex"})

        # Method of transforming LineIndex as a coordinate variable and reindexing the dataset onto it
        # Using .sel work but only for a single value of line_index as ther is
        # necessary duplicate in the coordinate variable.
        # This break the assumption that coordinate variables are monotic.
        # See: https://docs.xarray.dev/en/latest/user-guide/indexing.html
        # and: https://docs.unidata.ucar.edu/nug/current/netcdf_data_set_components.html#coordinate_variables
        # extracted_line = xr_ds_swapped.sel(LineIndex=line_index)

        # To select more than one line we can create a boolean mask for the line window
        # A bit slow
        mask = (xr_ds_swapped["LineIndex"] >= line_index) & (
            xr_ds_swapped["LineIndex"] <= line_index + line_window
        )
        extracted_line = xr_ds_swapped.where(mask, drop=True)

        extracted_line = extracted_line.to_dataframe()

        return extracted_line

    # def output_rgb(self):
    #     xr_ds = self.in_ds
    #
    #     import matplotlib.pyplot as plt
    #     from skimage import exposure
    #
    #     def adjust_gamma(img):
    #         corrected = exposure.adjust_gamma(img, 1)
    #         return corrected
    #
    #     def find_nearest(array, value):
    #         array = np.asarray(array)
    #         idx = (np.abs(array - value)).argmin()
    #         return idx
    #
    #     560 - xr_ds["radiance_at_sensor"].coords["wavelength"]
    #
    #     find_nearest(xr_ds["radiance_at_sensor"].coords["wavelength"], 650)
    #
    #     red_band = find_nearest(xr_ds["radiance_at_sensor"].coords["wavelength"], 650)
    #     green_band = find_nearest(xr_ds["radiance_at_sensor"].coords["wavelength"], 550)
    #     blue_band = find_nearest(xr_ds["radiance_at_sensor"].coords["wavelength"], 450)
    #
    #     red_array = xr_ds.isel(wavelength=red_band)["radiance_at_sensor"]
    #     green_array = xr_ds.isel(wavelength=green_band)["radiance_at_sensor"]
    #     blue_array = xr_ds.isel(wavelength=blue_band)["radiance_at_sensor"]
    #
    #     y_ticks, x_ticks = list(range(0, self.n_rows, 1000)), list(
    #         range(0, self.n_cols, 1000)
    #     )
    #     cor_x, cor_y = helper.transform_xy(self.Affine, rows=y_ticks, cols=x_ticks)
    #
    #     rgb_data = np.zeros((red_array.shape[0], red_array.shape[1], 3), dtype=float)
    #     rgb_data[:, :, 0] = red_array
    #     rgb_data[:, :, 1] = green_array
    #     rgb_data[:, :, 2] = blue_array
    #     dst = adjust_gamma(rgb_data)
    #     dst[~self.get_valid_mask()] = 1.0
    #     # dst = rgb_data
    #     plt.imshow(dst)
    #     plt.xticks(ticks=x_ticks, labels=cor_x)
    #     plt.yticks(ticks=y_ticks, labels=cor_y, rotation=90)
    #     plt.show()
