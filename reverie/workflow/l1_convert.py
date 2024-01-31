import os
import reverie.converter.wise as wise
from tqdm import tqdm
import numpy as np


def l1_convert(bundle_path: str = None, sensor: str = None):
    # TODO: l1_convert should find the appropriate parsing class from sensor.detect_bundle

    l1 = wise.read_pix.Pix(
        image_dir="/D/Data/TEST/TEST_WISE/MC-50/",
        image_name="190818_MC-50A-WI-2x1x1_v02",
    )
    l1.create_reve_nc(out_file=os.path.join(l1.image_dir, l1.image_name) + "-L1C.nc")

    # Create radiometric variable
    l1.create_var_nc(
        var="Lt",
        dimensions=(
            "W",
            "Y",
            "X",
        ),
    )

    for band in tqdm(range(0, l1.n_bands, 1), desc="Writing band: "):
        # GDAL use 1 base index
        data = l1.src_ds.GetRasterBand(band + 1)
        data = data.ReadAsArray()
        data = data * l1.scale_factor

        # Assign missing value
        data[data == 0] = l1.no_data * l1.scale_factor

        l1.nc_ds.variables["Lt"][band, :, :] = data

    # Create geometric variables
    geom = {
        "SolAzm": l1.solar_azimuth,
        "SolZen": l1.solar_zenith,
        "ViewAzm": l1.view_azimuth,
        "ViewZen": l1.viewing_zenith,
        "RelativeAzimuth": l1.relative_azimuth,
        "SampleIndex": l1.sample_index,
    }

    for var in tqdm(geom, desc="Writing geometry"):
        l1.create_var_nc(
            var=var,
            dimensions=(
                "Y",
                "X",
            ),
        )

        geom[var][np.isnan(geom[var])] = l1.no_data * l1.scale_factor

        l1.nc_ds.variables[var][:, :] = geom[var]

    l1.nc_ds.close()
    return
