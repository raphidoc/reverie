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
        local_dir=None,
        kind='linear',
        keep_series=False):

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
        local_files.append(download(anc_file))

    ## find if we have merra2 files
    gmao_files = [file for file in local_files if ('GMAO_MERRA2' in os.path.basename(file)) & (os.path.exists(file))]
    if len(gmao_files) == 0: gmao_files = [file for file in local_files if ('GMAO_FP' in os.path.basename(file)) & (os.path.exists(file))]
    if len(gmao_files) == 2:

        print('Using GMAO GEOS ancillary data:')
        for file in gmao_files: print(file)

        ## set up ancillary
        anc = {'date':dt, 'lon':lon, 'lat': lat, 'ftime':ftime, 'type': 'merra2', 'data': {}}
        anc_gmao = interp_gmao(gmao_files,  lon, lat, isodate, method=kind)
        for k in anc_gmao.keys():
            if (not keep_series) & ('series' in anc_gmao[k]): del anc_gmao[k]['series']
            anc['data'][k] = anc_gmao[k]

    anc_keys = None
    anc_fact = None

    ## rescale ancillary data
    anc_name = ['uoz', 'uwv', 'z_wind', 'm_wind', 'pressure']
    if anc['type'] == 'ncep':
        anc_keys = ['ozone', 'p_water', 'z_wind', 'm_wind', 'press']
        anc_fact = [1./1000., 1./10., 1., 1., 1.]
    elif anc['type'] == 'merra2':
        anc_keys = ['TO3', 'TQV', 'U10M', 'V10M', 'PS']
        anc_fact = [1./1000., 1./10., 1., 1., 1./100.]

    for i, k in enumerate(anc_keys):
        if k not in anc['data']: continue
        anc[anc_name[i]] = anc['data'][k]['interp'] * anc_fact[i]
    ## compute wind speed
    if ('z_wind' in anc) & ('m_wind' in anc):
        anc['wind'] = (((anc['z_wind'])**2) + ((anc['m_wind'])**2))**0.5

    return anc
