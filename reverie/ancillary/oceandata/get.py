"""
downloads and interpolates ancillary data from the ocean data server
"""
import datetime

from .parse_filename import parse_filename
from .download import download
from .interp_gmao import interp_gmao


def get(dt: datetime.datetime, lon, lat, local_dir=None):
    isodate = dt.isoformat()

    if isodate < "1978-10-27":
        print(f"Scene too old to get ancillary data: {isodate}")
        return

    # list ancillary files to download
    anc_files = parse_filename(dt)

    # download files
    local_files = []
    for anc_file in anc_files:
        local_files.append(download(anc_file, local_dir))

    print("Using GMAO GEOS ancillary data %s:" % local_files)

    anc_gmao = interp_gmao(local_files, lon, lat, dt)

    # scale ancillary data
    # anc_name = ['uoz', 'uwv', 'z_wind', 'm_wind', 'pressure']
    # anc_keys = ['TO3', 'TQV', 'U10M', 'V10M', 'PS']
    # anc_fact = [1. / 1000., 1. / 10., 1., 1., 1. / 100.]

    return anc_gmao
