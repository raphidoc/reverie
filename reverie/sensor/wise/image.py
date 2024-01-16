# Standard library imports
import os

# Third party imports
from osgeo import gdal
import netCDF4
from tqdm import tqdm
import numpy as np
import pyproj
import re
import pendulum

# REVERIE import
from reverie.image import Image
from .flightline import FlightLine
from reverie.utils import helper
from reverie.utils.tile import Tile

gdal.UseExceptions()


class PixWISE(Image):
    """
    This class expand Image() to read the WISE images in .pix format, compute the observation geometry and convert that data to the NetCDF CF
     reverie format.

    Image with extension .pix (PCIGeomatica) cand be read with the GDAL driver PCIDISK
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

        super().__init__()
        self.ImageDir = image_dir
        self.ImageName = image_name

        if os.path.isdir(image_dir):
            self.ImageDir = image_dir
        else:
            raise ValueError("image_dir does not exist")

        # WISE radiometric data
        self.hdr_f = os.path.join(image_dir, image_name + '-L1G.pix.hdr')
        self.pix_f = os.path.join(image_dir, image_name + '-L1G.pix')

        if not os.path.isfile(self.hdr_f) or not os.path.isfile(self.hdr_f):
            print(f"error: {self.hdr_f} or {self.pix_f} does not exist")

        # Parse ENVI header
        self.header = helper.read_envi_hdr(hdr_f=self.hdr_f)

        # Open the .pix file with GDAL
        self.src_ds = gdal.Open(self.pix_f)
        print(f"Dataset open with GDAL driver: {self.src_ds.GetDriver().ShortName}")

        # Define image cube size
        # self.src_ds.RasterYSize
        # Vertical axis: lines = rows = y
        # self.src_ds.RasterXSize
        # Horizontal axis: samples = columns = x
        self.n_rows = int(self.header['lines'])
        self.n_cols = int(self.header['samples'])
        self.n_bands = int(self.header['bands'])
        self.wavelength = np.array([float(w) for w in self.header['wavelength'].split(',')])

        # Define data type, unit, scale factor, offset, and ignore values
        dtype = int(self.header['data type'])
        if dtype == 1:
            self.dtype = np.dtype('uint8')
        if dtype == 2:
            self.dtype = np.dtype('int16')

        # TODO: how to deal with unit parsing ? Should we at all ? Just ask the user ?
        self.unit = 'uW cm-2 nm-1 sr-1'
        unit_res = {key: val for key, val in self.header.items() if re.search(f"unit", key)}
        print(f"Found unit: {unit_res}\nHard coded unit is: {self.unit}")
        #
        # if len(res) == 0:
        #     print("No unit found in the header you should probably: 1. yell of despair and 2. Try your best guess and contact someone :)")
        # if len(res) > 1:
        #     print("Multiple unit found")

        # either 'data scale factor' or 'radiance scale factor' can exist in ENVI hdr
        # self.scale_factor = int(self.Header.metadata['radiance scale factor'])
        #scale_res = {key: val for key, val in self.header.items() if re.search(f"scale", key)}
        self.scale_factor = int([val for key, val in self.header.items() if 'scale' in key][0])

        ignore_value = self.header['data ignore value']
        if ignore_value == '':
            self.no_data = -99999
        else:
            self.no_data = int(ignore_value)

        # Define image coordinate system from 'map info'
        map_info = self.header['map info']
        self.Affine, self.Proj, self.Proj4String = helper.parse_mapinfo(map_info)\

        # When in debugging mode with pdb, have to pass class and self to super(PixWISE, self)
        # see https://stackoverflow.com/questions/53508770/python-3-runtimeerror-super-no-arguments
        # TODO: Not even needed to call super().cal_coordinate ?
        self.cal_coordinate(self.Affine, self.n_rows, self.n_cols, self.Proj)

        # Define time for the image
        self.acq_time_z = pendulum.parse(self.header['acquisition time'], )

        self.cal_time(self.central_lon, self.central_lat)

        # Geocorrection Look Up tables
        self.glu_hdr_f = os.path.join(image_dir, image_name + '-L1A.glu.hdr')
        self.glu_f = os.path.join(image_dir, image_name + '-L1A.glu')

        # Navigation data: altitude, heading, pitch, roll, speed
        self.nav_f = os.path.join(image_dir, image_name + '-Navcor_sum.log')

        if not os.path.isfile(self.glu_hdr_f) or not os.path.isfile(self.glu_f) or not os.path.isfile(self.nav_f):
            print("Navigation data or geo correction missing, cannot compute viewing geometry.")
            self.flightline = None
            self.glu_f = None
            self.glu_hdr_f = None

        else:
            self.flightline = FlightLine.FromWISEFile(nav_sum_log=self.nav_f, glu_hdr=self.glu_hdr_f)


        # Other instance attribute that need to be instanced before population by methods
        self._valid_mask = None


        self.NetDS = None

        self.cal_viewing_geo()

        self.cal_sun_geom()

    def cal_viewing_geo(self):
        '''
        extract viewing zenith angle
        the orginal data from the flight line is not georeferenced, and the nrows and ncols are not the same as the georeferenced ones
        so, we need to transfer the original viewing geometry to the georefernce grid using the georeference LUT
        :return:
        '''
        if self.flightline is None:
            raise Exception(message='no flight line found')

        glu_data = None
        if self.glu_f is not None:
            glu_data = gdal.Open(self.glu_f)
            nchannels, nsamples_glu, nlines_glu = glu_data.RasterCount, glu_data.RasterXSize, glu_data.RasterYSize
            if nchannels != 3:
                raise Exception(message='the glu file does not have three channels')
            if nsamples_glu != self.flightline.samples or nlines_glu != self.flightline.lines:
                raise Exception(message='samples or lines of flightline and glu do not match')

        # data_glu =  glu_data.ReadAsArray()

        band_x, band_y = glu_data.GetRasterBand(1), glu_data.GetRasterBand(2)
        x_glu, y_glu = band_x.ReadAsArray(), band_y.ReadAsArray()

        v_zenith_fl, v_azimuth_fl = self.flightline.calSZA_AZA()
        # v_zenith_fl, v_azimuth_fl = v_zenith_fl.flatten(), v_azimuth_fl.flatten()

        ## initialize viewing zenith and azimuth with default values
        v_zenith_level1, v_azimuth_level1 = np.full((self.n_rows, self.n_cols), np.nan), np.full((self.n_rows, self.n_cols),
                                                                                                 np.nan)
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
        for row in tqdm(range(self.flightline.lines), desc='Processing GLU'):
            xs, ys = x_glu[row], y_glu[row]
            # print(xs,ys)
            rows_c, cols_c = helper.transform_rowcol(self.Affine, xs=xs, ys=ys, precision=5)
            mask = (rows_c < self.n_rows) & (cols_c < self.n_cols)
            # print(np.max(rows_c),np.max(cols_c))
            rows_c = rows_c[mask]
            cols_c = cols_c[mask]

            v_zenith_level1[rows_c, cols_c] = v_zenith_fl[row][mask]
            v_azimuth_level1[rows_c, cols_c] = v_azimuth_fl[row][mask]

            # Accros-track samples position on the spatial dimension of the imaging spectrometer array
            v_sample_array[rows_c, cols_c] = np.arange(0,self.flightline.samples)[mask]

        self.ViewingZenith = helper.FillNA2D(v_zenith_level1)
        self.view_azimuth = helper.FillNA2D(v_azimuth_level1)
        self.SampleIndex = helper.FillNA2D(v_sample_array)

        self.ViewingZenith[~self.get_valid_mask()] = np.nan
        self.view_azimuth[~self.get_valid_mask()] = np.nan
        self.SampleIndex[~self.get_valid_mask()] = np.nan

    def to_netcdf(self, outfile=None, var=None, unit=None, savegeometry = True, compression="zlib", complevel=1):

        if outfile is None:
            outfile = os.path.join(self.ImageDir, self.ImageName) + '-L1C.nc'

        if not var is None:
            var = var
        else:
            var = self.VarName

        if not unit is None:
            unit = unit
        else:
            unit = self.unit

        # if Var == None:
        #     print('Writing varibles: '+ )

        width, height, bands = self.n_cols, self.n_rows, self.n_bands
        wavelengths = self.wavelength
        transform = self.Affine

        # try:
        #     NetDS = netCDF4.Dataset(OutFile, "w", format="NETCDF4")
        # except Exception as e:
        #     print("Cannot create: "+OutFile)
        #     raise e
        # else:

        with netCDF4.Dataset(outfile, "w", format="NETCDF4") as NetDS:

            # Global Attributes
            # TODO validate that it follow the convention with cfdm
            NetDS.Conventions = 'CF-1.10'
            NetDS.title = self.ImageName
            NetDS.acquisition_time = self.acq_time_z.strftime('%y-%m-%dT%H:%M:%SZ')
            NetDS.history = 'File created on ' + pendulum.now("utc").strftime('%y-%m-%dT%H:%M:%SZ')
            NetDS.institution = 'AquaTel UQAR'
            #NetDS.source = 'ac4icw'
            #NetDS.version = '0.1.0'
            #NetDS.references = 'Ask Yanqun Pan'
            NetDS.comment = 'Conversion from pix to NetCDF'

            # Set coordinates and CRS attributes

            x1, y1 = helper.transform_xy(transform,
                                  rows=0,
                                  cols=range(0, width, 1),
                                  offset='center')

            x2, y2 = helper.transform_xy(transform,
                                  rows=range(0, height, 1),
                                  cols=0,
                                  offset='center')

            NetDS.createDimension('y', height)
            NetDS.createDimension('x', width)
            y_var = NetDS.createVariable('y', 'f4', ('y',))
            x_var = NetDS.createVariable('x', 'f4', ('x',))

            y_var[:] = y2
            x_var[:] = x1

            y_var.units = 'm'
            y_var.standard_name = 'projection_y_coordinate'
            y_var.long_name = 'y-coordinate in projected coordinate system'
            y_var.axis = 'y'

            x_var.units = 'm'
            x_var.standard_name = 'projection_x_coordinate'
            x_var.long_name = 'x-coordinate in projected coordinate system'
            x_var.axis = 'x'

            # TODO proj4 result in loss of projection information, see: https://proj4.org/faq.html#what-is-the-best-format-for-describing-coordinate-reference-systems
            crs = pyproj.CRS.from_proj4(self.Proj4String)
            print('Detected EPSG:'+str(crs.to_epsg()))
            crs = pyproj.CRS.from_epsg(crs.to_epsg())
            cf_grid_mapping = crs.to_cf()

            ProjVar = NetDS.createVariable(cf_grid_mapping['grid_mapping_name'], np.int32, ())

            ProjVar.crs_wtk = cf_grid_mapping['crs_wkt']
            ProjVar.semi_major_axis = cf_grid_mapping['semi_major_axis']
            ProjVar.semi_minor_axis = cf_grid_mapping['semi_minor_axis']
            ProjVar.inverse_flattening = cf_grid_mapping['inverse_flattening']
            ProjVar.reference_ellipsoid_name = cf_grid_mapping['reference_ellipsoid_name']
            ProjVar.longitude_of_prime_meridian = cf_grid_mapping['longitude_of_prime_meridian']
            ProjVar.prime_meridian_name = cf_grid_mapping['prime_meridian_name']
            ProjVar.geographic_crs_name = cf_grid_mapping['geographic_crs_name']
            ProjVar.horizontal_datum_name = cf_grid_mapping['horizontal_datum_name']
            ProjVar.projected_crs_name = cf_grid_mapping['projected_crs_name']
            ProjVar.grid_mapping_name = cf_grid_mapping['grid_mapping_name']
            ProjVar.latitude_of_projection_origin = cf_grid_mapping['latitude_of_projection_origin']
            ProjVar.longitude_of_central_meridian = cf_grid_mapping['longitude_of_central_meridian']
            ProjVar.false_easting = cf_grid_mapping['false_easting']
            ProjVar.false_northing = cf_grid_mapping['false_northing']
            ProjVar.scale_factor_at_central_meridian = cf_grid_mapping['scale_factor_at_central_meridian']

            # Not sure that it's the right place to save that
            #ProjVar.geotransform = Transform

            ### wavelength coordinates

            NetDS.createDimension('band', bands)
            band_var = NetDS.createVariable('band', 'f4', ('band',))
            band_var[:] = wavelengths #([round(w, 2) for w in wavelengths])

            band_var.units = 'nm'
            band_var.standard_name = 'sensor_band_central_radiation_wavelength'
            band_var.axis = 'B'

            ### Data Variable

            # np.iinfo(type)
            # np.finfo(type)

            data_var = NetDS.createVariable(
                varname='Lt',
                datatype='i4',
                dimensions=('band', 'y', 'x',),
                fill_value=self.no_data,
                compression=compression,
                complevel=complevel)#,
                #significant_digits=5)

            # Should follow the standard name table CF convention
            data_var.units = 'uW-1cm-2sr-1nm-1'
            data_var.standard_name = 'Lt'
            data_var.long_name = 'at sensor radiance'

            data_var.grid_mapping = ProjVar.name

            # Set attributes for the Rrs variable
            #self.__dst.variables['Rrs'].valid_min = 0
            #self.__dst.variables['Rrs'].valid_max = 6000
            data_var.missing_value = self.no_data

            '''
            scale_factor is used by NetCDF CF in writing and reading
            Reading: multiply by the scale_factor and add the add_offset
            Writing: subtract the add_offset and divide by the scale_factor
            If the scale factor is integer, to properly apply the scale_factor in the writing order we need the
            reciprocal of it.
            '''
            scale_factor = np.reciprocal(float(self.scale_factor))
            data_var.scale_factor = scale_factor
            data_var.add_offset = 0

            #BandI = 3

        # Try to use multiprocessing to speed up the IO, maybe asyncio would be more appropriate ?
        # bands = list(range(1, self.n_bands))
        #
        # src_ds = self.src_ds
        #
        # #args = (self.pix_f, outfile, self.no_data, scale_factor, height, width, bands)
        #
        # args = [(self.pix_f, outfile, self.no_data, scale_factor, height, width, band) for band in bands]
        #
        # with Pool() as pool:
        #     results = pool.map(pool_write_nc, args)

            for BandI in tqdm(range(0, self.n_bands, 1), desc='Writing band: '):
                # print('Writing band: '+ str(BandI))

                # GDAL use 1 base index
                BandTemp = self.src_ds.GetRasterBand(BandI + 1)
                Data = BandTemp.ReadAsArray()
                Data = Data.astype('f4') * scale_factor

                # Data = Data/float(Ed0[BandI][0])

                # Assing missing value
                Data[Data == 0] = self.no_data * scale_factor

                # print("NetCDF variable shape: "+str(NetDS.variables['Lt'].shape))
                # print("Matrix shape: "+str(Lt.shape))
                # NetCDF use 0 base index
                NetDS.variables['Lt'][BandI, 0:height, 0:width] = Data

            if savegeometry:

                sol_azm = self.solar_azimuth
                sol_zen = self.solar_zenith
                view_azm = self.view_azimuth
                view_zen = self.ViewingZenith

                geom = {'SolAzm': sol_azm, 'SolZen': sol_zen, 'ViewAzm': view_azm, 'ViewZen': view_zen}

                for x in tqdm(geom.keys(), desc='Writing geometry'):
                    geometry_var = NetDS.createVariable(
                        varname=x,
                        datatype='f4',
                        dimensions=('y', 'x',),
                        fill_value=self.no_data,
                        compression='zlib',
                        complevel=1)  # ,
                    # significant_digits=5)

                    # Should follow the standard name table CF convention
                    geometry_var.units = 'degree'
                    geometry_var.standard_name = x
                    #DataVar.long_name = 'at sensor radiance'

                    geometry_var.grid_mapping = ProjVar.name

                    # Set attributes for the Rrs variable
                    # self.__dst.variables['Rrs'].valid_min = 0
                    # self.__dst.variables['Rrs'].valid_max = 6000
                    geometry_var.missing_value = self.no_data

                    '''
                    scale_factor is used by NetCDF CF in writing and reading
                    Reading: multiply by the scale_factor and add the add_offset
                    Writing: subtract the add_offset and divide by the scale_factor
                    If the scale factor is integer, to properly apply the scale_factor in the writing order we need the
                    reciprocal of it.
                    '''
                    # For Geometry type the scale_factor is set to keep a 6 digit precision
                    #scale_factor = 1e-6 #np.reciprocal(float(self.scale_factor))
                    #GeometryVar.scale_factor = scale_factor
                    #GeometryVar.add_offset = 0

                    geom[x][np.isnan(geom[x])] = self.no_data #* scale_factor

                    NetDS.variables[x][0:height, 0:width] = geom[x]

                # Save the sample position on the spatial dimension of the imaging spectrometer array
                sample_index = self.SampleIndex

                sample_var = NetDS.createVariable(
                    varname='SampleIndex',
                    datatype='i4',
                    dimensions=('y', 'x',),
                    fill_value=self.no_data,
                    compression='zlib',
                    complevel=1)  # ,
                # significant_digits=5)

                # Should follow the standard name table CF convention
                sample_var.units = 'degree'
                sample_var.standard_name = x
                # DataVar.long_name = 'at sensor radiance'

                sample_var.grid_mapping = ProjVar.name

                # Set attributes for the Rrs variable
                # self.__dst.variables['Rrs'].valid_min = 0
                # self.__dst.variables['Rrs'].valid_max = 6000
                sample_var.missing_value = self.no_data

                sample_index[np.isnan(sample_index)] = self.no_data  # * scale_factor

                NetDS.variables['SampleIndex'][0:height, 0:width] = sample_index

            #NetDS.close()
