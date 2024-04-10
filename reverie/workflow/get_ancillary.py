from reverie import ReveCube
from reverie.ancillary import oceandata


def ac_run(bundle_path: str = None, sensor: str = None):
    # l1_convert(bundle_path)

    l1 = ReveCube.from_reve_nc(
        in_path="/D/Data/WISE/MC-37A/190818_MC-37A-WI-1x1x1_v02-L1CG.nc"
    )

    dt = l1.acq_time_z
    central_lon = l1.lon[round(len(l1.lon) / 2)]
    central_lat = l1.lat[round(len(l1.lat) / 2)]
    print(
        f"Getting ancillary data for {dt.strftime('%Y-%m-%d %H:%M:%SZ')} {central_lon:.3f}E {central_lat:.3f}N"
    )
    anc = oceandata.get(
        dt, central_lon, central_lat, local_dir="/D/Data/TEST/anc"
    )


if __name__ == "__main__":
    ac_run("/D/Data/TEST/TEST_WISE/MC-50/")
