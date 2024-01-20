""" downloads and interpolates ancillary data from the ocean data server
written by Quinten Vanhellemont, RBINS for the PONDER project (2017-10-18)
modified by Raphael Mabit (2024-01-18)
"""

import os
import datetime

from .parse_filename import parse_filename
from .download import download
from .interp_gmao import interp_gmao


def get(dt: datetime.datetime,
        lon, lat,
        local_dir=None):

    isodate = dt.isoformat() #'2019-08-18T17:34:56+00:00'
    ftime = dt.hour + dt.minute/60 + dt.second/3600

    # diff = (dateutil.parser.parse(datetime.datetime.now().strftime('%Y-%m-%d')) -
    #         dateutil.parser.parse(dt.strftime('%Y-%m-%d'))).days
    # if diff < 7:
    #     print('Scene too recent to get ancillary data: {}'.format(isodate))
    #     return

    if isodate < '1978-10-27':
        print(f'Scene too old to get ancillary data: {isodate}')
        return

    # list ancillary files

    anc_files = parse_filename(dt)

    # download files
    local_files = []
    for anc_file in anc_files:
        local_files.append(download(anc_file, local_dir))

    ## find if we have 2 merra2 files
    if len(local_files) == 2:

        print('Using GMAO GEOS ancillary data %s:' %local_files)

        anc_gmao = interp_gmao(local_files,  lon, lat, dt)

    # scale ancillary data
    anc_name = ['uoz', 'uwv', 'z_wind', 'm_wind', 'pressure']
    anc_keys = ['TO3', 'TQV', 'U10M', 'V10M', 'PS']
    anc_fact = [1./1000., 1./10., 1., 1., 1./100.]


    return anc
