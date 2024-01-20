## interp_gmao
## interpolates GMAO GEOS data from hourly files to given lon, lat and time (float)
##
## written by Quinten Vanhellemont, RBINS
## 2023-10-16
## modifications:

import os, sys
import numpy as np
from datetime import datetime
from scipy import interpolate

import netCDF4
import xarray as xr

from reverie.image.revecube import ReveCube


def interp_gmao(files, lon, lat, dt):
    """
    Interpolates GMAO GEOS data from hourly files to image  lon, lat and time
    Parameters
    ----------
    files
    lon
    lat
    dt

    Returns
    -------

    """

    # requested date/time
    ftime = dt.hour + dt.minute/60 + dt.second/3600

    def preprocess_gmao(ds):
        """
        add time coordinate to gmao dataset
        Parameters
        ----------
        ds

        Returns
        -------

        """
        ds = ds.assign_coords({'T': datetime.strptime(ds.time_coverage_start, "%Y-%m-%dT%H:%M:%SZ").timestamp()})
        ds = ds.expand_dims('T')
        return ds

    net_ds = xr.open_mfdataset(files, preprocess=preprocess_gmao, parallel=False)

    gmao_interp = net_ds.interp(lat=lat, lon=lon, T=dt.timestamp())
    gmao_interp = gmao_interp.compute()

    # interpolate for the whole image
    # interpolated = da.interp_like(image)

    return gmao_interp
