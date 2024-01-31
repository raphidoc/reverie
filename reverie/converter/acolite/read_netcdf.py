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
import pyproj

# REVERIE import
from reverie.image import ReveCube

gdal.UseExceptions()


class AcoliteNetCDF(ReveCube):
    def __init__(self, nc_file):
        if os.path.isfile(nc_file):
            src_ds = netCDF4.Dataset(nc_file, "r", format="NETCDF4")

        else:
            raise Exception(f"File {nc_file} does not exist")

        # ACOLITE store spectral variable in a wide format (individual varaible by wavelength)
        # and wavelength in variables name and attributes
        # First list all variable that are wavelength dependant
        pattern = re.compile(r".*_\d{3}")

        def find_matching_keys(pattern, dictionary):
            return list(filter(lambda key: re.search(pattern, key), dictionary.keys()))

        spectral_var = find_matching_keys(pattern, src_ds.variables)

        # Create a list of wavelength from the variable attributes
        wavelength = []
        for var in spectral_var:
            wavelength.append(src_ds.variables[var].wavelength)

        # Convert list to numpy array
        wavelength = np.round(np.array(wavelength), 2)

        # Create a dictionary mapping variable name to wavelength
        self.spectral_var = dict(zip(spectral_var, wavelength))

        # Geographic attributes
        # As ACOLITE only process satellite imagery we give a symbolic altitude of 800km
        z = 800  # As we have only one altitude, could be a scalar

        grid_mapping = src_ds.variables["transverse_mercator"]
        crs = pyproj.CRS.from_wkt(grid_mapping.crs_wkt)
        affine = None
        n_rows = src_ds.dimensions["y"].size
        n_cols = src_ds.dimensions["x"].size

        x_var = src_ds.variables["x"]
        y_var = src_ds.variables["y"]
        lon_var = src_ds.variables["lon"]
        lat_var = src_ds.variables["lat"]
        x = x_var[:].data
        y = y_var[:].data
        # ACOLITE lon and lat have both y, x dimension as in CF-1.11 section 5.2
        # don't know why this was done
        lon = lon_var[:, 0].data
        lat = lat_var[0, :].data
        lon_grid, lat_grid = None, None
        center_lon, center_lat = None, None

        # Time attributes
        time_var = src_ds.isodate

        acq_time_z = datetime.fromisoformat(time_var)
        acq_time_local = None, None
        central_lon_local_timezone = None

        super().__init__(
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

    def to_reve_nc(self, out_file=None):
        """
        Convert ACOLITE NetCDF to REVE NetCDF
        """
        if out_file is None:
            out_file = f"{os.path.splitext(self.src_ds.filepath())[0]}-reve.nc"

        # Create REVE NetCDF
        self.create_reve_nc(out_file)

        # Create radiometric variable
        # ACOLITE doesn't use a scale factor
        self.create_var_nc(
            var="rhow",
            datatype="f4",
            dimensions=(
                "W",
                "Y",
                "X",
            ),
            scale_factor=1,
        )

        for var_name, wavelength in tqdm(
            self.spectral_var.items(), desc="Writing band: "
        ):
            data = self.src_ds.variables[var_name][:, :].data
            # Cannot directly index with wavelength value as to find the index of the wavelength
            # we need to round the value as the conversion to list appear to modify it
            wave_ix = np.where(self.nc_ds.variables["W"][:].data == wavelength)[0][0]
            # Could also just assume the order is correct and use the index of the variable
            self.nc_ds.variables["rhow"][wave_ix, :, :] = data

        # # Create geometric variables
        # geom = {
        #     "SolAzm": self.solar_azimuth,
        #     "SolZen": self.solar_zenith,
        #     "ViewAzm": self.view_azimuth,
        #     "ViewZen": self.viewing_zenith,
        #     "RelativeAzimuth": self.relative_azimuth,
        #     "SampleIndex": self.sample_index,
        # }
        #
        # for var in tqdm(geom, desc="Writing geometry"):
        #     self.create_var_nc(
        #         var=var,
        #         dimensions=(
        #             "Y",
        #             "X",
        #         ),
        #     )
        #
        #     geom[var][np.isnan(geom[var])] = self.no_data * self.scale_factor
        #
        #     self.nc_ds.variables[var][:, :] = geom[var]

        self.nc_ds.close()
        return
