# Standard library imports
import os
import time
from datetime import datetime

# Third party imports
from osgeo import gdal
from tqdm import tqdm
import numpy as np
import re
import netCDF4

# REVERIE import
from reverie.image import ReveCube
from .flightline import FlightLine
from reverie.utils import helper

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
        self.hdr_f = os.path.join(image_dir, image_name + "-L1G.pix.hdr")
        self.pix_f = os.path.join(image_dir, image_name + "-L1G.pix")

        if not os.path.isfile(self.hdr_f) or not os.path.isfile(self.pix_f):
            print(f"error: {self.hdr_f} or {self.pix_f} does not exist")

        self.image_name = image_name

        # Parse ENVI header
        self.header = helper.read_envi_hdr(hdr_f=self.hdr_f)

        # Open the .pix file with GDAL
        self.src_ds = gdal.Open(self.pix_f)
        print(f"Dataset open with GDAL driver: {self.src_ds.GetDriver().ShortName}")

        # Define image cube size
        # self.src_ds.RasterYSize
        # Vertical axis: lines = rows = height = y
        # self.src_ds.RasterXSize
        # Horizontal axis: samples = columns = width = x
        self.n_rows = int(self.header["lines"])
        self.n_cols = int(self.header["samples"])
        self.n_bands = int(self.header["bands"])
        self.wavelength = np.array(
            [float(w) for w in self.header["wavelength"].split(",")]
        )

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

        # Define image coordinate system from 'map info'
        map_info = self.header["map info"]
        self.Affine, self.CRS, self.Proj4String = helper.parse_mapinfo(map_info)
        # When in debugging mode with pdb, have to pass class and self to super(Pix, self)
        # see https://stackoverflow.com/questions/53508770/python-3-runtimeerror-super-no-arguments
        self.cal_coordinate(self.Affine, self.n_rows, self.n_cols, self.CRS)

        # Define time for the image
        # datetime.strptime("21/11/06T16:30:00Z", "%d/%m/%yT%H:%M:%SZ")
        self.acq_time_z = datetime.strptime(
            self.header["acquisition time"], "%Y-%m-%dT%H:%M:%SZ"
        )

        self.cal_time(self.center_lon, self.center_lat)

        # Geocorrection Look Up tables
        self.glu_hdr_f = os.path.join(image_dir, image_name + "-L1A.glu.hdr")
        self.glu_f = os.path.join(image_dir, image_name + "-L1A.glu")

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

        # Other instance attribute that need to be instanced before population by methods
        self._valid_mask = None

        self.cal_sun_geom()

        self.cal_view_geom()

        self.cal_relative_azimuth()

        # TODO: Need to learn more about super(), inheritance and composition.
        super().__init__(
            src_ds=self.src_ds,
            wavelength=self.wavelength,
            z=self.z,
            affine=self.Affine,
            n_rows=self.n_rows,
            n_cols=self.n_cols,
            crs=self.CRS,
            x=self.x,
            y=self.y,
            lon=self.lon,
            lat=self.lat,
            lon_grid=self.lon_grid,
            lat_grid=self.lat_grid,
            center_lon=self.center_lon,
            center_lat=self.center_lat,
            acq_time_z=self.acq_time_z,
            acq_time_local=self.acq_time_local,
            central_lon_local_timezone=self.central_lon_local_timezone,
        )

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
            if nchannels != 3:
                raise Exception("the glu file does not have three channels")
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
        v_zenith_level1, v_azimuth_level1 = np.full(
            (self.n_rows, self.n_cols), np.nan
        ), np.full((self.n_rows, self.n_cols), np.nan)
        # Initiate the sample position on spatial dimension of the imaging spectrometer array
        v_sample_array = np.full((self.n_rows, self.n_cols), np.nan)

        # self._XY()
        # print('looking for correlated corrdinates')
        # for row in tqdm(range(self.nrows),desc='Processing GLU'):
        #     y =  self.y[row]
        #     temp_y =  np.abs(y_glu-y)
        #
        #     for col in tqdm(range(self.ncols),desc='Line {}'.format(row)):
        #         x = self.x[col]
        #         min_index = np.argmin(np.abs(x_glu-x)+temp_y)
        #         v_zenith_level1[row,col] = v_zenith_fl[min_index]
        #         v_azimuth_level1[row,col] =  v_azimuth_fl[min_index]
        #
        # del v_zenith_fl, v_azimuth_fl,x_glu,y_glu
        # self.viewing_zenith, self.viewing_azimuth =  v_zenith_level1, v_azimuth_level1
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

    def to_reve_nc(self):
        """
        Convert the Pix() object to reve NetCDF CF format
        :return:
        """
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

        for band in tqdm(range(0, self.n_bands, 1), desc="Writing band: "):
            # GDAL use 1 base index
            data = self.src_ds.GetRasterBand(band + 1)
            data = data.ReadAsArray()
            data = data * self.scale_factor

            # Assign missing value
            # -2147483647 is the default no data value for int32
            """
            scale_factor is used by NetCDF CF in writing and reading
            Reading: multiply by the scale_factor and add the add_offset
            Writing: subtract the add_offset and divide by the scale_factor
            If the scale factor is integer, to properly apply the scale_factor in the writing order we need the
            reciprocal of it.
            """
            data[data == 0] = self.no_data * self.scale_factor

            self.nc_ds.variables["Lt"][band, :, :] = data

        # Create geometric variables
        geom = {
            "SolAzm": self.solar_azimuth,
            "SolZen": self.solar_zenith,
            "ViewAzm": self.view_azimuth,
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

            self.nc_ds.variables[var][:, :] = geom[var]

        self.nc_ds.close()
        return
