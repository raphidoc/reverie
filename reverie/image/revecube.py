# Standard library imports
import os
from abc import ABC
import datetime
import re

# Third party imports
import netCDF4
import numpy as np
import pyproj
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from p_tqdm import p_uimap
import shapely
import xarray as xr
import math

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
    def from_reve_nc(cls, src_file):
        """Populate ReveCube object from reve CF NetCDF dataset

        Parameters
        ----------
        src_file: str
            NetcCDF (.nc) CF-1.0 compliant file to read from

        Returns
        -------
        """

        if os.path.isfile(src_file):
            src_ds = netCDF4.Dataset(src_file, "r", format="NETCDF4")
        else:
            raise Exception(f"File {src_file} does not exist")

        # Spectral attributes
        wavelength_var = src_ds.variables["W"]
        wavelength = wavelength_var[:].data

        # Spatiotemporal attributes
        time_var = src_ds.variables["T"]
        acq_time_z = datetime.datetime.fromtimestamp(time_var[:][0])
        altitude_var = src_ds.variables["Z"]
        z = altitude_var[:][0]
        y = src_ds.variables["Y"][:].data
        x = src_ds.variables["X"][:].data
        lat = src_ds.variables["lat"][:].data
        lon = src_ds.variables["lon"][:].data
        n_rows = src_ds.dimensions["Y"].size
        n_cols = src_ds.dimensions["X"].size

        grid_mapping = src_ds.variables["grid_mapping"]
        affine = None
        crs = pyproj.CRS.from_wkt(grid_mapping.crs_wkt)

        return cls(
            src_file,
            src_ds,
            wavelength,
            acq_time_z,
            z,
            y,
            x,
            lat,
            lon,
            n_rows,
            n_cols,
            affine,
            crs,
        )

    def __init__(
        self,
        src_file: str,
        src_ds: netCDF4.Dataset,
        wavelength: np.ndarray,
        acq_time_z,
        z: float,
        y: np.ndarray,
        x: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        n_rows: int,
        n_cols: int,
        affine,
        crs: pyproj.crs.CRS,
    ):
        # Dataset accessor
        self.in_file = src_file
        self.in_ds = src_ds

        # Spectral attributes
        self.wavelength = wavelength

        # Spatiotemporal attributes
        self.acq_time_z = acq_time_z
        self.z = z
        self.y = y
        self.x = x
        self.lon = lon
        self.lat = lat
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.Affine = affine
        self.CRS = crs

        # Sensor attribute store metadata from the sensor in a dictionary
        self.sensor = {}

        # Additional attributes
        self.out_file = None
        self.out_ds = None

        self.no_data = None
        self._valid_mask = None
        # self.proj_var = None

        # Geographic attributes
        self.lon_grid, self.lat_grid = None, None
        self.center_lon, self.center_lat = None, None

        # Time attributes
        self.acq_time_local = None
        self.central_lon_local_timezone = None

        # Geometric attributes
        self.solar_zenith = None
        self.solar_azimuth = None
        self.viewing_zenith = None
        self.view_azimuth = None
        self.relative_azimuth = None

        # Pixel location on the sensor array
        self.sample_index = None

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

        offset_hours = self.acq_time_local.utcoffset().total_seconds() / 3600

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
        band_temp = self.in_ds.GetRasterBand(bandindex + 1)
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
        self.out_ds = nc_ds

    def create_var_nc(
        self,
        var: str = None,
        datatype=None,
        dimensions: tuple = None,
        scale_factor: float = 1.0,
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

        ds = self.out_ds

        # Easier to leave missing_value as the default _FillValue,
        # but then GDAL doesn't recognize it ...
        # self.no_data = netCDF4.default_fillvals[datatype]
        # When scaling the default _FillValue, it get somehow messed up when reading with GDAL
        self.no_data = math.trunc(netCDF4.default_fillvals[datatype] * 0.00001)

        data_var = ds.createVariable(
            varname=var,
            datatype=datatype,
            dimensions=dimensions,
            fill_value=self.no_data,
            compression=compression,
            complevel=complevel,
        )

        data_var.grid_mapping = "grid_mapping"  # self.proj_var.name

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
        data_var.missing_value = self.no_data

        """
        scale_factor is used by NetCDF CF in writing and reading
        Reading: multiply by the scale_factor and add the add_offset
        Writing: subtract the add_offset and divide by the scale_factor
        If the scale factor is integer, to properly apply the scale_factor in the writing order we need the
        reciprocal of it.
        """
        data_var.scale_factor = scale_factor
        data_var.add_offset = 0

    def to_reve_nc(self):
        """
        TODO : Wrapper for create_dataset_nc, create_var_nc, write_var_nc, close_nc ?
        Returns
        -------
        """
        pass

    def extract_pixel(
        self, matchup_file: str = None, var_name: list = None, window_size: int = 1
    ):
        """
        Method to exctract the pixel matchups provided by `matchup_gdf`
        Should provide a way to select variables to be extracted
        Another method should list the variable that can be extracted
        Should also add some metadata from the CF convention to the output like Acquisition time, Processing (atmcor), ...
        """

        # load the nc dataset with xarray
        xr_ds = xr.open_dataset(self.in_file)

        matchup_gdf = pd.read_csv(matchup_file)
        matchup_gdf.columns = matchup_gdf.columns.str.lower()
        matchup_gdf = matchup_gdf[["datetime", "lon", "lat", "uuid"]]

        # When matchup data is in long format
        matchup_gdf = matchup_gdf.drop_duplicates()

        match_geometry = gpd.points_from_xy(
            matchup_gdf["lon"], matchup_gdf["lat"], crs="EPSG:4326"
        )

        matchup_gdf = gpd.GeoDataFrame(matchup_gdf, geometry=match_geometry)

        print(f"Projecting in-situ data to EPSG: {self.CRS.to_epsg()}")
        matchup_gdf = matchup_gdf.to_crs(self.CRS.to_epsg())

        # TODO manage the variables to be extracted, either read from all available variable or user input
        #   Take a look at the method isel_window(window: Window, pad: bool = False)

        pixex_df = pd.DataFrame()

        # def extractor(uuid, out_ds=self.out_ds, matchup_gdf=matchup_gdf):

        # iterator = p_uimap(extractor, matchup_gdf["UUID"])

        pixex_df = pd.DataFrame()
        for uuid in tqdm(matchup_gdf["uuid"]):
            temp_gdf = matchup_gdf[matchup_gdf["uuid"] == uuid].reset_index()

            temp_pix_ex_array = xr_ds.sel(
                X=shapely.get_x(temp_gdf.geometry)[0],
                Y=shapely.get_y(temp_gdf.geometry)[0],
                # isodate=pd.to_datetime(temp_gdf['DateTime'][0]),
                method="nearest",
            )

            # For some reason the Sensor variable create an error with to_dataframe().
            # Seems to be linked to the data type (string, <U7 or object)
            # temp_pixex_df = temp_pix_ex_array.to_array(name='Values')
            # temp_pixex_df = temp_pix_ex_array.to_dataframe(name='Values')
            temp_pixex_df = temp_pix_ex_array.to_dataframe()
            # temp_pixex_df = temp_pixex_df.rename_axis("Wavelength")
            # temp_pixex_df = temp_pixex_df.reset_index()
            # TODO output a wide format when wavelength and non wavelength data are mixed
            # temp_pixex_df = pd.pivot(temp_pixex_df, index=['x', 'y'], columns='Wavelength', values='Lt')
            temp_pixex_df["uuid"] = uuid
            # temp_pixex_df['Sensor'] = str(temp_pix_ex_array.Sensor.values)
            # temp_pixex_df['ImageDate'] = str(temp_pix_ex_array.coords['isodate'].values)
            # temp_pixex_df['AtCor'] = 'ac4icw'

            pixex_df = pd.concat([pixex_df, temp_pixex_df], axis=0)

        return pixex_df
