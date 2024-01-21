from reverie import ReveCube
from reverie.ancillary import oceandata
from l1_convert import l1_convert


def ac_run(bundle_path: str = None, sensor: str = None):

    #l1_convert(bundle_path)

    l1 = ReveCube.from_nc(nc_file="/D/Data/TEST/TEST_WISE/MC-50/190818_MC-50A-WI-2x1x1_v02-L1C.nc")

    dt = l1.acq_time_z
    central_lon = l1.center_lon
    central_lat = l1.center_lat
    print(f"Getting ancillary data for {dt.strftime('%Y-%m-%d %H:%M:%SZ')} {central_lon:.3f}E {central_lat:.3f}N")
    anc = oceandata.get(dt, central_lon, central_lat)


if __name__ == '__main__':
    ac_run("/D/Data/TEST/TEST_WISE/MC-50/")
