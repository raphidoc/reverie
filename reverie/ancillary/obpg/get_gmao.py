"""
downloads and interpolates ancillary data from the ocean data server
"""
import os

from patsy.state import center

from reverie.ancillary.obpg.interp_gmao import interp_gmao
from reverie.ancillary.obpg import OBPGSession

import datetime

import xarray as xr
import numpy as np

PATH_TO_DATA = "/D/Data/TEST/"


def get_gmao(l1):
    server = "oceandata.sci.gsfc.nasa.gov"

    if not os.path.exists(os.path.join(PATH_TO_DATA, "anc")):
        os.makedirs(os.path.join(PATH_TO_DATA, "anc"))

    dt = l1.acq_time_z

    l1.expand_coordinate()
    center_lon = l1.center_lon
    center_lat = l1.center_lat
    print(
        f"Getting ancillary data for {dt.strftime('%Y-%m-%dT%H:%M:%SZ')} {center_lon:.3f}E {center_lat:.3f}N"
    )

    # Select .MET files hour before and after observation time
    met_files = []
    yyyymmdd = dt.strftime("%Y%m%d")
    hh = str(dt.hour).zfill(2)
    cfile = f"GMAO_MERRA2.{yyyymmdd}T{hh}0000.MET.nc"
    met_files.append(cfile)

    # Handle change of day to select hour after observation time
    if hh < "23":
        hh = str(dt.hour + 1).zfill(2)
        cfile = f"GMAO_MERRA2.{yyyymmdd}T{hh}0000.MET.nc"
        met_files.append(cfile)
    else:
        yyyymmdd = (dt + datetime.timedelta(days=1)).strftime("%Y%m%d")
        hh = "00"
        cfile = f"GMAO_MERRA2.{yyyymmdd}T{hh}0000.MET.nc"
        met_files.append(cfile)

    met_local_files = []
    ancPath = os.path.join(PATH_TO_DATA, "anc")

    met1 = met_files[0]
    met1_path = os.path.join(PATH_TO_DATA, "anc", met1)

    if not os.path.exists(met1_path):
        # request = f"/cgi/getfile/{met1}"
        request = f"/ob/getfile/{met1}"
        msg = f"Retrieving ancillary file from server: {met1}"
        print(msg)

        status = OBPGSession.httpdl(
            server,
            request,
            localpath=ancPath,
            outputfilename=met1,
            uncompress=False,
            verbose=2,
        )
    else:
        status = 200
        msg = f"Ancillary file found locally: {met1}"
        print(msg)

    met2 = met_files[1]
    met2_path = os.path.join(PATH_TO_DATA, "anc", met2)

    if not os.path.exists(met2_path):
        # request = f"/cgi/getfile/{met2}"
        request = f"/ob/getfile/{met2}"
        msg = f"Retrieving anchillary file from server: {met2}"
        print(msg)

        status = OBPGSession.httpdl(
            server,
            request,
            localpath=ancPath,
            outputfilename=met2,
            uncompress=False,
            verbose=2,
        )
    else:
        status = 200
        msg = f"Ancillary file found locally: {met2}"
        print(msg)

    if status in (400, 401, 403, 404, 416):
        msg = f"Request error: {status}"
        print(msg)

        return None

    met_local_files = [met1_path, met2_path]

    # Select hour before observation time
    aer_files = []
    yyyymmdd = dt.strftime("%Y%m%d")
    hh = str(dt.hour).zfill(2)
    cfile = f"GMAO_MERRA2.{yyyymmdd}T{hh}0000.AER.nc"
    aer_files.append(cfile)

    # Handle change of day to select hour after observation time
    if hh < "23":
        hh = str(dt.hour + 1).zfill(2)
        cfile = f"GMAO_MERRA2.{yyyymmdd}T{hh}0000.AER.nc"
        aer_files.append(cfile)
    else:
        yyyymmdd = (dt + datetime.timedelta(days=1)).strftime("%Y%m%d")
        hh = "00"
        cfile = f"GMAO_MERRA2.{yyyymmdd}T{hh}0000.AER.nc"
        aer_files.append(cfile)

    aer_local_files = []
    ancPath = os.path.join(PATH_TO_DATA, "anc")

    aer1 = aer_files[0]
    aer1_path = os.path.join(PATH_TO_DATA, "anc", aer1)

    if not os.path.exists(aer1_path):
        # request = f"/cgi/getfile/{met1}"
        request = f"/ob/getfile/{aer1}"
        msg = f"Retrieving anchillary file from server: {aer1}"
        print(msg)

        status = OBPGSession.httpdl(
            server,
            request,
            localpath=ancPath,
            outputfilename=aer1,
            uncompress=False,
            verbose=2,
        )
    else:
        status = 200
        msg = f"Ancillary file found locally: {aer1}"
        print(msg)

    aer2 = aer_files[1]
    aer2_path = os.path.join(PATH_TO_DATA, "anc", aer2)

    if not os.path.exists(aer2_path):
        # request = f"/cgi/getfile/{met2}"
        request = f"/ob/getfile/{aer2}"
        msg = f"Retrieving anchillary file from server: {aer2}"
        print(msg)

        status = OBPGSession.httpdl(
            server,
            request,
            localpath=ancPath,
            outputfilename=aer2,
            uncompress=False,
            verbose=2,
        )
    else:
        status = 200
        msg = f"Ancillary file found locally: {aer2}"
        print(msg)

    if status in (400, 401, 403, 404, 416):
        msg = f"Request error: {status}"
        print(msg)

        return None

    aer_local_files = [aer1_path, aer2_path]

    # Total Aerosol Extinction AOT 550 nm, same as AOD(550)
    # ancTExt = aerGroup.getDataset("TOTEXTTAU")
    # ancTExt.attributes["wavelength"] = "550 nm"

    print(
        f"Using GMAO \n MET {met_local_files} \n AER {aer_local_files}"
    )

    met_anc = interp_gmao(met_local_files, center_lon, center_lat, dt)
    aer_anc = interp_gmao(aer_local_files, center_lon, center_lat, dt)

    anc = xr.merge([met_anc, aer_anc])

    return anc
