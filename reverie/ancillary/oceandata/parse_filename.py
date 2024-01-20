import os
import datetime, dateutil.parser


def parse_filename(dt: datetime.datetime, file_types=None):
    """
    Create the filenames bounding observation time to download from the OB.DAAC.
    Parameters
    ----------
    dt: datetime.datetime
        Image observation time
    file_types

    Returns
    -------
    anc_files: filenames bounding observation time
    """

    # get ancillary file types
    file_types = ["GMAO_MERRA2_MET"]  # ac.settings['run']['ancillary_type'] # GMAO_MERRA2_MET

    anc_files = []
    if "GMAO_MERRA2_MET" in file_types:
        # TODO: Select GMAO hourly file before and after image acquisition for interpolation on time coordinate
        yyyymmdd = dt.strftime('%Y%m%d')
        hh = str(dt.hour).zfill(2)
        cfile = f"GMAO_MERRA2.{yyyymmdd}T{hh}0000.MET.nc"
        anc_files.append(cfile)

        # Handle change of day to select next hour
        if hh < '23':
            hh = str(dt.hour+1).zfill(2)
            cfile = f"GMAO_MERRA2.{yyyymmdd}T{hh}0000.MET.nc"
            anc_files.append(cfile)
        else:
            yyyymmdd = (dt+datetime.timedelta(days=1)).strftime("%Y%m%d")
            hh = '00'
            cfile = f"GMAO_MERRA2.{yyyymmdd}T{hh}0000.MET.nc"
            anc_files.append(cfile)

    return anc_files
