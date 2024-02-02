from reverie import ReveCube
from reverie.converter import wise

if __name__ == "__main__":
    # l1 = wise.read_pix.Pix(
    #     image_dir="/D/Data/TEST/TEST_WISE/MC-51A/",
    #     image_name="190818_MC-51A-WI-2x1x1_v02",
    # )
    #
    # l1.to_reve_nc()

    reve_nc = ReveCube.from_reve_nc(
        "/D/Data/TEST/TEST_WISE/MC-51A/190818_MC-51A-WI-2x1x1_v02-L1C.nc"
    )

    test = reve_nc.extract_pixel(
        "/D/Documents/PhD/Thesis/Chapter2/Data/WISE/Match_MP_20190818.csv",
        var_name="Lt",
    )
