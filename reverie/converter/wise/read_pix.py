# Standard library imports
import os
import time
import datetime
import math

# Third party imports
from osgeo import gdal
from tqdm import tqdm
import numpy as np
import re
import netCDF4
from p_tqdm import p_uimap
import xarray as xr


# REVERIE import
from reverie.image import ReveCube
from .flightline import FlightLine
from reverie.utils import helper
from reverie.utils.tile import Tile
from reverie.utils.cf_aliases import get_cf_std_name

# from reverie.utils.tile import Tile

gdal.UseExceptions()


class Pix(ReveCube):
    """
    This class expand ReveCube() to read the WISE images in .pix format, compute the observation geometry and convert
     that data to the NetCDF CF reverie format.

    DateCube with extension .pix (PCIGeomatica) cand be read with the GDAL driver PCIDISK
    They are minimally are composed of two files:
    * .pix containing the image data
    * .pix.hdr ENVI header describing the .pix data

    Additionally, geo correction LUT might be present:
    * .glu contains the geolocation data of the pixels across the track of the plane
    * .glu.hdr ENVI header describing the .glu data

    Additionnally, navigation files might be present:
    * -Navcor_sum.log statistical summary of the NAVCOR.log
    * -NAVCOR.log: full navigation log from the GNSS IMU system
    """

    def __init__(self, image_dir, image_name):
        t0 = time.perf_counter()

        if os.path.isdir(image_dir):
            self.image_dir = image_dir
        else:
            raise ValueError("image_dir does not exist")

        # WISE radiometric data
        self.hdr_f = os.path.join(image_dir, image_name + "-L1CG.pix.hdr")
        self.pix_f = os.path.join(image_dir, image_name + "-L1CG.pix")

        if not os.path.isfile(self.hdr_f) or not os.path.isfile(self.pix_f):
            print(f"error: {self.hdr_f} or {self.pix_f} does not exist")

        self.image_name = image_name

        # Open the .pix file with GDAL
        self.src_ds = gdal.Open(self.pix_f)
        print(f"Dataset open with GDAL driver: {self.src_ds.GetDriver().ShortName}")

        # Parse ENVI header
        self.header = helper.read_envi_hdr(hdr_f=self.hdr_f)

        self.wavelength = np.array(
            [float(w) for w in self.header["wavelength"].split(",")]
        )

        # Define time for the image
        # datetime.strptime("21/11/06T16:30:00Z", "%d/%m/%yT%H:%M:%SZ")
        self.acq_time_z = datetime.datetime.strptime(
            self.header["acquisition time"], "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=datetime.timezone.utc)

        # Define image cube size
        # self.in_ds.RasterYSize
        # Vertical axis: lines = rows = height = y
        # self.in_ds.RasterXSize
        # Horizontal axis: samples = columns = width = x
        self.n_rows = int(self.header["lines"])
        self.n_cols = int(self.header["samples"])
        self.n_bands = int(self.header["bands"])

        # Define image coordinate from 'map info'
        map_info = self.header["map info"]
        self.Affine, self.CRS, self.Proj4String = helper.parse_mapinfo(map_info)
        # Compute projected and geographic coordinates
        self.cal_coordinate(self.Affine, self.n_rows, self.n_cols, self.CRS)

        # Define data type, unit, scale factor, offset, and ignore values
        dtype = int(self.header["data type"])
        if dtype == 1:
            self.dtype = np.dtype("uint8")
        if dtype == 2:
            self.dtype = np.dtype("int16")

        # TODO: how to deal with unit parsing ? Should we at all ? Just ask the user ?
        self.unit = "uW cm-2 nm-1 sr-1"
        unit_res = {
            key: val for key, val in self.header.items() if re.search(f"unit", key)
        }
        print(f"Found unit: {unit_res}\nHard coded unit is: {self.unit}")
        #
        # if len(res) == 0:
        #     print("No unit found in the header you should probably: 1. yell of despair and 2. Try your best guess and contact someone :)")
        # if len(res) > 1:
        #     print("Multiple unit found")

        # either 'data scale factor' or 'radiance scale factor' can exist in ENVI hdr
        # self.scale_factor = int(self.Header.metadata['radiance scale factor'])
        # scale_res = {key: val for key, val in self.header.items() if re.search(f"scale", key)}
        """
        scale_factor is used by NetCDF CF in writing and reading
        Reading: multiply by the scale_factor and add the add_offset
        Writing: subtract the add_offset and divide by the scale_factor
        If the scale factor is integer, to properly apply the scale_factor in the writing order we need the
        reciprocal of it.
        """
        scale_factor = [val for key, val in self.header.items() if "scale" in key][0]
        print(f"Read scale factor as: {scale_factor}")
        try:
            scale_factor = int(scale_factor)
            self.scale_factor = np.reciprocal(float(scale_factor))
        except ValueError as e:
            print(e)
            self.scale_factor = float(scale_factor)
        print(f"Converted scale factor as: {self.scale_factor}")
        ignore_value = self.header["data ignore value"]
        if ignore_value == "":
            self.no_data = -99999
        else:
            self.no_data = int(ignore_value)

        # Geocorrection Look Up tables
        # In case Algae-WISE, glu file are named L2C
        self.glu_hdr_f = os.path.join(image_dir, image_name + "-L2C.glu.hdr")
        self.glu_f = os.path.join(image_dir, image_name + "-L2C.glu")

        # Navigation data: altitude, heading, pitch, roll, speed
        self.nav_f = os.path.join(image_dir, image_name + "-Navcor_sum.log")

        if (
            not os.path.isfile(self.glu_hdr_f)
            or not os.path.isfile(self.glu_f)
            or not os.path.isfile(self.nav_f)
        ):
            print(
                "Navigation data or geo correction missing, cannot compute viewing geometry."
            )
            self.flightline = None
            self.glu_f = None
            self.glu_hdr_f = None

        else:
            self.flightline = FlightLine.from_wise_file(
                nav_sum_log=self.nav_f, glu_hdr=self.glu_hdr_f
            )
            self.z = self.flightline.height

        super().__init__(
            src_file=self.pix_f,
            src_ds=self.src_ds,
            wavelength=self.wavelength,
            acq_time_z=self.acq_time_z,
            z=self.z,
            y=self.y,
            x=self.x,
            lat=self.lat,
            lon=self.lon,
            n_rows=self.n_rows,
            n_cols=self.n_cols,
            affine=self.Affine,
            crs=self.CRS,
        )

        self.cal_coordinate_grid(self.lat, self.lon)

        self.cal_time(self.center_lon, self.center_lat)

        # Other instance attribute that need to be instanced before population by methods
        # self._valid_mask = None

        self.cal_sun_geom()

        self.cal_view_geom()

        self.cal_relative_azimuth()

        t1 = time.perf_counter()

        print(
            f"ReveCube initiated from class {self.__class__.__name__} in {t1-t0:.2f}s"
        )

    def cal_view_geom(self):
        """
        extract viewing zenith angle
        the orginal data from the flight line is not georeferenced, and the nrows and ncols are not the same as the georeferenced ones
        so, we need to transfer the original viewing geometry to the georefernce grid using the georeference LUT
        :return:
        """
        if self.flightline is None:
            raise Exception("no flight line found")

        glu_data = None
        if self.glu_f is not None:
            glu_data = gdal.Open(self.glu_f)
            nchannels, nsamples_glu, nlines_glu = (
                glu_data.RasterCount,
                glu_data.RasterXSize,
                glu_data.RasterYSize,
            )
            # WISEMan glu file has three channels but not AlgaeWISE
            # However, the first two channels are the same
            # if nchannels != 3:
            #     raise Exception("the glu file does not have three channels")
            if (
                nsamples_glu != self.flightline.samples
                or nlines_glu != self.flightline.lines
            ):
                raise Exception("samples or lines of flightline and glu do not match")

        # data_glu =  glu_data.ReadAsArray()

        band_x, band_y = glu_data.GetRasterBand(1), glu_data.GetRasterBand(2)
        x_glu, y_glu = band_x.ReadAsArray(), band_y.ReadAsArray()

        v_zenith_fl, v_azimuth_fl = self.flightline.cal_view_geom()
        # v_zenith_fl, v_azimuth_fl = v_zenith_fl.flatten(), v_azimuth_fl.flatten()

        ## initialize viewing zenith and azimuth with default values
        v_zenith_level1 = np.full((self.n_rows, self.n_cols), np.nan)
        v_azimuth_level1 = np.full((self.n_rows, self.n_cols), np.nan)
        # Initialize the sample position on spatial dimension of the imaging spectrometer array
        v_sample_array = np.full((self.n_rows, self.n_cols), np.nan)

        # TODO: use multiprocessing here
        for row in tqdm(range(self.flightline.lines), desc="Processing GLU"):
            xs, ys = x_glu[row], y_glu[row]
            # print(xs,ys)
            rows_c, cols_c = helper.transform_rowcol(
                self.Affine, xs=xs, ys=ys, precision=5
            )
            mask = (rows_c < self.n_rows) & (cols_c < self.n_cols)
            # print(np.max(rows_c),np.max(cols_c))
            rows_c = rows_c[mask]
            cols_c = cols_c[mask]

            v_zenith_level1[rows_c, cols_c] = v_zenith_fl[row][mask]
            v_azimuth_level1[rows_c, cols_c] = v_azimuth_fl[row][mask]

            # Accros-track samples position on the spatial dimension of the imaging spectrometer array
            v_sample_array[rows_c, cols_c] = np.arange(0, self.flightline.samples)[mask]

        self.viewing_zenith = helper.fill_na_2d(v_zenith_level1)
        self.view_azimuth = helper.fill_na_2d(v_azimuth_level1)
        self.sample_index = helper.fill_na_2d(v_sample_array)

        self.viewing_zenith[~self.get_valid_mask()] = np.nan
        self.view_azimuth[~self.get_valid_mask()] = np.nan
        self.sample_index[~self.get_valid_mask()] = np.nan

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
            # TODO: Problem when the band read is flag in as bbl (bad band list),
            #   the mask contains only False. Should filter out the bbl bands.
            iband = 10
            Lt = self.read_band(iband)
            self._valid_mask = Lt > 0

    def to_reve_nc(self):
        """
        Convert the Pix() object to reve NetCDF CF format
        :return:
        """

        # TODO: use p_uimap to create the NetCDF files band by band with multiple threads and then merge them.
        #  The function works when callded with a single band but fails when called with multiple threads (bands).
        #  Get TypeError: cannot pickle 'SwigPyObject' object.
        #  Is this because we are accessing the same GDAL dataset with multiple threads ?
        # def multi_thread_create_reve_nc(band):
        #     out_file = (
        #         f"{os.path.join(self.image_dir, 'L1_bands', 'band'+str(band))}-L1C.nc"
        #     )
        #
        #     try:
        #         nc_ds = netCDF4.Dataset(out_file, "w", format="NETCDF4")
        #     except Exception as e:
        #         print(e)
        #         return
        #
        #     # TODO validate that it follow the convention with cfdm / cf-python.
        #     #  For compatibility with GDAL NetCDF driver use CF-1.0
        #     nc_ds.Conventions = "CF-1.0"
        #     nc_ds.title = "Remote sensing image written by REVERIE"
        #     nc_ds.history = "File created on " + datetime.datetime.utcnow().strftime(
        #         "%Y-%m-%d %H:%M:%SZ"
        #     )
        #     nc_ds.institution = "AquaTel UQAR"
        #     nc_ds.source = "Remote sensing imagery"
        #     nc_ds.version = "0.1.0"
        #     nc_ds.references = "https://github.com/raphidoc/reverie"
        #     nc_ds.comment = "Reflectance Extraction and Validation for Environmental Remote Imaging Exploration"
        #
        #     # Create Dimensions
        #     nc_ds.createDimension("W", len([self.wavelength[band]]))
        #     nc_ds.createDimension("T", len([self.acq_time_z]))
        #     nc_ds.createDimension("Z", len([self.z]))
        #     nc_ds.createDimension("Y", len(self.y))
        #     nc_ds.createDimension("X", len(self.x))
        #
        #     band_var = nc_ds.createVariable("W", "f4", ("W",))
        #     band_var.units = "nm"
        #     band_var.standard_name = "radiation_wavelength"
        #     band_var.long_name = "Central wavelengths of the converter bands"
        #     band_var.axis = "Wavelength"
        #     band_var[:] = self.wavelength[band]
        #
        #     # Create coordinate variables
        #     # We will store time as seconds since 1 january 1970 good luck people of 2038 :) !
        #     t_var = nc_ds.createVariable("T", "f4", ("T",))
        #     t_var.standard_name = "time"
        #     t_var.long_name = "UTC acquisition time of remote sensing image"
        #     # CF convention for time zone is UTC if ommited
        #     # xarray will convert it to a datetime64[ns] object considering it is local time
        #     t_var.units = "seconds since 1970-01-01 00:00:00 +00:00"
        #     t_var.calendar = "gregorian"
        #     t_var[:] = self.acq_time_z.timestamp()
        #
        #     z_var = nc_ds.createVariable("Z", "f4", ("Z",))
        #     z_var.units = "m"
        #     z_var.standard_name = "altitude"
        #     z_var.long_name = (
        #         "Altitude is the viewing height above the geoid, positive upward"
        #     )
        #     z_var.axis = "y"
        #     z_var[:] = self.z
        #
        #     y_var = nc_ds.createVariable("Y", "f4", ("Y",))
        #     y_var.units = "m"
        #     y_var.standard_name = "projection_y_coordinate"
        #     y_var.long_name = "y-coordinate in projected coordinate system"
        #     y_var.axis = "y"
        #     y_var[:] = self.y
        #
        #     lat_var = nc_ds.createVariable("lat", "f4", ("Y",))
        #     lat_var.standard_name = "latitude"
        #     lat_var.units = "degrees_north"
        #     # lat_var.long_name = 'latitude'
        #     lat_var[:] = self.lat
        #
        #     x_var = nc_ds.createVariable("X", "f4", ("X",))
        #     x_var.units = "m"
        #     x_var.standard_name = "projection_x_coordinate"
        #     x_var.long_name = "x-coordinate in projected coordinate system"
        #     x_var.axis = "x"
        #     x_var[:] = self.x
        #
        #     lon_var = nc_ds.createVariable("lon", "f4", ("X",))
        #     lon_var.standard_name = "longitude"
        #     lon_var.units = "degrees_east"
        #     # lon_var.long_name = 'longitude'
        #     lon_var[:] = self.lon
        #
        #     # grid_mapping
        #     crs = self.CRS
        #     # print("Detected EPSG:" + str(crs.to_epsg()))
        #     cf_grid_mapping = crs.to_cf()
        #
        #     grid_mapping = nc_ds.createVariable("grid_mapping", np.int32, ())
        #
        #     grid_mapping.grid_mapping_name = cf_grid_mapping["grid_mapping_name"]
        #     grid_mapping.crs_wkt = cf_grid_mapping["crs_wkt"]
        #     grid_mapping.semi_major_axis = cf_grid_mapping["semi_major_axis"]
        #     grid_mapping.semi_minor_axis = cf_grid_mapping["semi_minor_axis"]
        #     grid_mapping.inverse_flattening = cf_grid_mapping["inverse_flattening"]
        #     grid_mapping.reference_ellipsoid_name = cf_grid_mapping[
        #         "reference_ellipsoid_name"
        #     ]
        #     grid_mapping.longitude_of_prime_meridian = cf_grid_mapping[
        #         "longitude_of_prime_meridian"
        #     ]
        #     grid_mapping.prime_meridian_name = cf_grid_mapping["prime_meridian_name"]
        #     grid_mapping.geographic_crs_name = cf_grid_mapping["geographic_crs_name"]
        #     grid_mapping.horizontal_datum_name = cf_grid_mapping[
        #         "horizontal_datum_name"
        #     ]
        #     grid_mapping.projected_crs_name = cf_grid_mapping["projected_crs_name"]
        #     grid_mapping.grid_mapping_name = cf_grid_mapping["grid_mapping_name"]
        #     grid_mapping.latitude_of_projection_origin = cf_grid_mapping[
        #         "latitude_of_projection_origin"
        #     ]
        #     grid_mapping.longitude_of_central_meridian = cf_grid_mapping[
        #         "longitude_of_central_meridian"
        #     ]
        #     grid_mapping.false_easting = cf_grid_mapping["false_easting"]
        #     grid_mapping.false_northing = cf_grid_mapping["false_northing"]
        #     grid_mapping.scale_factor_at_central_meridian = cf_grid_mapping[
        #         "scale_factor_at_central_meridian"
        #     ]
        #
        #     # Easier to leave missing_value as the default _FillValue,
        #     # but then GDAL doesn't recognize it ...
        #     # self.no_data = netCDF4.default_fillvals[datatype]
        #     # When scaling the default _FillValue, it get somehow messed up when reading with GDAL
        #     self.no_data = math.trunc(netCDF4.default_fillvals["i4"] * 0.00001)
        #
        #     data_var = nc_ds.createVariable(
        #         varname="Lt",
        #         datatype="i4",
        #         dimensions=("W", "Y", "X"),
        #         fill_value=self.no_data,
        #         compression="zlib",
        #         complevel=1,
        #     )
        #
        #     data_var.grid_mapping = "grid_mapping"  # self.proj_var.name
        #
        #     # Follow the standard name table CF convention
        #     std_name, std_unit = get_cf_std_name(alias="Lt")
        #
        #     data_var.units = std_unit
        #     data_var.standard_name = std_name
        #     # data_var.long_name = ''
        #
        #     # self.__dst.variables['Rrs'].valid_min = 0
        #     # self.__dst.variables['Rrs'].valid_max = 6000
        #     data_var.missing_value = self.no_data
        #
        #     """
        #     scale_factor is used by NetCDF CF in writing and reading
        #     Reading: multiply by the scale_factor and add the add_offset
        #     Writing: subtract the add_offset and divide by the scale_factor
        #     If the scale factor is integer, to properly apply the scale_factor in the writing order we need the
        #     reciprocal of it.
        #     """
        #     data_var.scale_factor = self.scale_factor
        #     data_var.add_offset = 0
        #
        #     # GDAL use 1 base index
        #     data = self.src_ds.GetRasterBand(band)
        #     data = data.ReadAsArray()
        #     data = data * self.scale_factor
        #
        #     # Assign missing value
        #     data[data == 0] = self.no_data * self.scale_factor
        #
        #     nc_ds.variables["Lt"][:, :, :] = data
        #
        #     nc_ds.close()
        #
        #     pass
        #
        # # multi_thread_create_reve_nc(7)
        #
        # iterator = p_uimap(multi_thread_create_reve_nc, range(1, 2, 1))
        #
        # for result in enumerate(iterator):
        #     continue
        #
        # test = xr.open_mfdataset(
        #     os.path.join(self.image_dir, self.image_name, "L1_bands")
        # )

        # Create NetCDF file
        self.create_reve_nc(
            out_file=f"{os.path.join(self.image_dir, self.image_name)}-L1C.nc"
        )

        # Create radiometric variable
        self.create_var_nc(
            var="Lt",
            datatype="i4",
            dimensions=(
                "W",
                "Y",
                "X",
            ),
            scale_factor=self.scale_factor,
        )
        # self.n_bands
        for band in tqdm(range(0, self.n_bands, 1), desc="Writing band: "):
            # GDAL use 1 base index
            # Reading from GDAL can be realy slow
            data = self.src_ds.GetRasterBand(band + 1)
            data = data.ReadAsArray()
            data = data * self.scale_factor

            # Assign missing value
            """
            scale_factor is used by NetCDF CF in writing and reading
            Reading: multiply by the scale_factor and add the add_offset
            Writing: subtract the add_offset and divide by the scale_factor
            If the scale factor is integer, to properly apply the scale_factor in the writing order we need the
            reciprocal of it.
            """
            data[data == 0] = self.no_data * self.scale_factor

            self.out_ds.variables["Lt"][band, :, :] = data

        # Create geometric variables
        geom = {
            "SolAzi": self.solar_azimuth,
            "SolZen": self.solar_zenith,
            "ViewAzi": self.view_azimuth,
            "ViewZen": self.viewing_zenith,
            "RelativeAzimuth": self.relative_azimuth,
            "SampleIndex": self.sample_index,
        }

        for var in tqdm(geom, desc="Writing geometry"):
            self.create_var_nc(
                var=var,
                datatype="i4",
                dimensions=(
                    "Y",
                    "X",
                ),
                scale_factor=self.scale_factor,
            )
            data = geom[var]

            np.nan_to_num(data, copy=False, nan=self.no_data * self.scale_factor)

            self.out_ds.variables[var][:, :] = data

        self.out_ds.close()
        return
