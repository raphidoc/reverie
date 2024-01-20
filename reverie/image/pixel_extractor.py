import os
import xarray as xr
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import shapely
import pyproj


class PixelExtractor():
    def __init__(self):

        if os.path.isfile(NCFilePath):
            self.NetDS = xr.open_dataset(NCFilePath, mask_and_scale=True)
        else:
            raise ValueError('File not found')

        DataVars = list(self.NetDS.data_vars)

        # The first variable is the proj_var
        self.CRS = pyproj.CRS.from_wkt(self.NetDS[DataVars[0]].crs_wtk)

        # The rest is the actual data
        self.VarName = DataVars[1:]

    def assing_matchup(self, MatchupF, Vars=None):
        '''
        Method to read a data frame with potential matchups to be extracted by `exctract_pixel`
        Data Frame is read as csv (sep = , decimal = .)
        Must have the column "DateTime", "Lon", "Lat", "UUID"
        '''

        MatchDF = pd.read_csv(MatchupF)
        MatchDF = MatchDF[["DateTime", "Lon", "Lat", "UUID"]]

        # When matchup data is in long format
        MatchDF = MatchDF.drop_duplicates()

        Geometry = gpd.points_from_xy(MatchDF['Lon'], MatchDF['Lat'], crs="EPSG:4326")

        MatchGDF = gpd.GeoDataFrame(MatchDF, geometry=Geometry)

        print(f"Projecting in-situ data to EPSG: {self.CRS.to_epsg()}")
        MatchGDF = MatchGDF.to_crs(self.CRS.to_epsg())

        self.MatchGDF = MatchGDF

        # print('Removing in-situ observations outside of true image extent (excluding NoData)')
        #
        # NetF=os.path.join(self.ImageDir, self.ImageName) + '_L1CG.nc'
        # print('Looking for NetCDF image: '+NetF)
        # if os.path.isfile(NetF):
        #     NetDS = xarray.open_dataset(NetF, mask_and_scale=False)
        # else:
        #     print('File not found, run `to_netcdf()` first.')

        # CFDS = xr.decode_cf(NetDS)

        # Variable 0 is the transverse mercator proj
        # Overlay = NetDS.isel(band=0).to_array()[1]
        # Contour = xr.plot.contour(Overlay, levels=1)
        # fig, ax = plt.subplots()
        # CS = ax.contour(Overlay, levels=1)
        # ax.clabel(CS, inline=True, fontsize=10)
        # ax.set_title('Simplest default with labels')
        # plt.show()
        #
        # fig, ax = plt.subplots()
        #
        # polygon = Polygon(Contour.collections[0].get_paths()[4].vertices)
        # Points = MatchupGDF.geometry
        #
        # #print(polygon.contains(Points))
        #
        # xs = [point.x for point in Points]
        # ys = [point.y for point in Points]
        #
        # x, y = polygon.exterior.xy
        # #ax.plot(x, y, c="red")
        # ax.scatter(xs, ys)
        # plt.show()

    def extract_pixel(self, Vars=None):
        '''
        Method to exctract the pixel matchups defined by `assigned_matchup`
        Should provide a way to select variables to be extracted
        Another method should list the variable that can be extracted
        Should also add some metadata from the CF convention to the output like Acquisition time, Processing (atmcor), ...
        '''

        # TODO manage the variables to be extracted, either read from all available variable or user input
        if Vars == None:
            Vars = self.VarName

        NetDS = self.NetDS
        MatchGDF = self.MatchGDF

        MSIPixEx = pd.DataFrame()

        # x = self.MatchGDF['UUID'][0]

        for x in tqdm(MatchGDF['UUID'], desc='Extracting pixels'):
            TempGDF = MatchGDF[MatchGDF['UUID'] == x].reset_index()
            TempPixExArray = NetDS.sel(
                x=shapely.get_x(TempGDF.geometry)[0],
                y=shapely.get_y(TempGDF.geometry)[0],
                # isodate=pd.to_datetime(TempGDF['DateTime'][0]),
                method='nearest')

            # For some reason the Sensor variable create an error with to_dataframe().
            # Seems to be linked to the data type (string, <U7 or object)
            # TempPixExDF = TempPixExArray.to_array(name='Values')
            # TempPixExDF = TempPixExArray.to_dataframe(name='Values')
            TempPixExDF = TempPixExArray.to_dataframe()
            TempPixExDF = TempPixExDF.rename_axis('Wavelength')
            TempPixExDF = TempPixExDF.reset_index()
            # TODO output a wide format when wavelength and non wavelength data are mixed
            #TempPixExDF = pd.pivot(TempPixExDF, index=['x', 'y'], columns='Wavelength', values='Lt')
            TempPixExDF['UUID'] = x
            # TempPixExDF['Sensor'] = str(TempPixExArray.Sensor.values)
            # TempPixExDF['ImageDate'] = str(TempPixExArray.coords['isodate'].values)
            # TempPixExDF['AtCor'] = 'ac4icw'

            MSIPixEx = pd.concat([MSIPixEx, TempPixExDF])

        # MSIPixEx.columns = ['_'.join(str(col)).strip() for col in MSIPixEx.columns.values]
        # MSIPixEx.reset_index(inplace=True)
        #MSIPixEx = pd.DataFrame(MSIPixEx.to_records())
        #MSIPixEx.columns = MSIPixEx.columns.str.replace(r"\(|'|\)|,|\s(?!\d)", "", regex=True)
        #MSIPixEx.columns = MSIPixEx.columns.str.replace(r" ", "_", regex=True)
        return MSIPixEx

        # Take a look at the method isel_window(window: Window, pad: bool = False)