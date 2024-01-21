import os
import xarray as xr
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import shapely
import pyproj


class PixelExtractor():
    def __init__(self, nc_file: str):

        if os.path.isfile(nc_file):
            self.NetDS = xr.open_dataset(nc_file)
        else:
            raise ValueError('File not found')

        self.match_gdf = None

        data_vars = list(self.NetDS.data_vars)

        # The first variable is the proj_var
        self.CRS = pyproj.CRS.from_wkt(self.NetDS[data_vars[0]].crs_wtk)

        # The rest is the actual data
        self.VarName = data_vars[1:]

    def assing_matchup(self, matchup_file: str):
        '''
        Method to read a data frame with potential matchups to be extracted by `exctract_pixel`
        Data Frame is read as csv (sep = , decimal = .)
        Must have the column "DateTime", "Lon", "Lat", "UUID"
        '''

        match_df = pd.read_csv(matchup_file)
        match_df = match_df[["DateTime", "Lon", "Lat", "UUID"]]

        # When matchup data is in long format
        match_df = match_df.drop_duplicates()

        match_geometry = gpd.points_from_xy(match_df['Lon'], match_df['Lat'], crs="EPSG:4326")

        match_gdf = gpd.GeoDataFrame(match_df, geometry=match_geometry)

        print(f"Projecting in-situ data to EPSG: {self.CRS.to_epsg()}")
        match_gdf = match_gdf.to_crs(self.CRS.to_epsg())

        self.match_gdf = match_gdf

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

    def extract_pixel(self):
        '''
        Method to exctract the pixel matchups defined by `assigned_matchup`
        Should provide a way to select variables to be extracted
        Another method should list the variable that can be extracted
        Should also add some metadata from the CF convention to the output like Acquisition time, Processing (atmcor), ...
        '''

        # TODO manage the variables to be extracted, either read from all available variable or user input
        #   Take a look at the method isel_window(window: Window, pad: bool = False)

        net_ds = self.NetDS
        match_gdf = self.match_gdf

        pixex_df = pd.DataFrame()

        # x = self.match_gdf['UUID'][0]

        for x in tqdm(match_gdf['UUID'], desc='Extracting pixels'):
            temp_gdf = match_gdf[match_gdf['UUID'] == x].reset_index()
            temp_pix_ex_array = net_ds.sel(
                x=shapely.get_x(temp_gdf.geometry)[0],
                y=shapely.get_y(temp_gdf.geometry)[0],
                # isodate=pd.to_datetime(temp_gdf['DateTime'][0]),
                method='nearest')

            # For some reason the Sensor variable create an error with to_dataframe().
            # Seems to be linked to the data type (string, <U7 or object)
            # temp_pixex_df = temp_pix_ex_array.to_array(name='Values')
            # temp_pixex_df = temp_pix_ex_array.to_dataframe(name='Values')
            temp_pixex_df = temp_pix_ex_array.to_dataframe()
            temp_pixex_df = temp_pixex_df.rename_axis('Wavelength')
            temp_pixex_df = temp_pixex_df.reset_index()
            # TODO output a wide format when wavelength and non wavelength data are mixed
            # temp_pixex_df = pd.pivot(temp_pixex_df, index=['x', 'y'], columns='Wavelength', values='Lt')
            temp_pixex_df['UUID'] = x
            # temp_pixex_df['Sensor'] = str(temp_pix_ex_array.Sensor.values)
            # temp_pixex_df['ImageDate'] = str(temp_pix_ex_array.coords['isodate'].values)
            # temp_pixex_df['AtCor'] = 'ac4icw'

            pixex_df = pd.concat([pixex_df, temp_pixex_df])

        # pixex_df.columns = ['_'.join(str(col)).strip() for col in pixex_df.columns.values]
        # pixex_df.reset_index(inplace=True)
        # pixex_df = pd.DataFrame(pixex_df.to_records())
        # pixex_df.columns = pixex_df.columns.str.replace(r"\(|'|\)|,|\s(?!\d)", "", regex=True)
        # pixex_df.columns = pixex_df.columns.str.replace(r" ", "_", regex=True)
        return pixex_df
