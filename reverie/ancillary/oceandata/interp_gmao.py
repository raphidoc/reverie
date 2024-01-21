from datetime import datetime

import xarray as xr


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

    def prepro_gmao(ds):
        """
        add time coordinate to gmao dataset
        Parameters
        ----------
        ds

        Returns
        -------

        """
        ds = ds.assign_coords(
            {'T': datetime.strptime(ds.time_coverage_start, "%Y-%m-%dT%H:%M:%SZ").timestamp()}
        )
        ds = ds.expand_dims('T')
        return ds

    net_ds = xr.open_mfdataset(files, preprocess=prepro_gmao, parallel=False)

    gmao_interp = net_ds.interp(lat=lat, lon=lon, T=dt.timestamp())
    gmao_interp = gmao_interp.compute()

    # interpolate for the whole image
    # interpolated = da.interp_like(image)

    return gmao_interp
