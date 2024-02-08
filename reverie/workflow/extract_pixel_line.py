from reverie import ReveCube
from reverie.converter import wise

if __name__ == "__main__":
    l1 = wise.read_pix.Pix(
        pix_dir="/D/Data/TEST/TEST_WISE/MC-51A/190818_MC-51A-WI-2x1x1_v02-L1CG.dpix"
    )

    l1.to_reve_nc()

    # reve = ReveCube.from_zarr(
    #     "/D/Data/TEST/TEST_WISE/MC-10A/190820_MC-10A-WI-1x1x1_v02-L1C.zarr"
    # )

    # test = reve.extract_pixel_line(
    #     "/D/Documents/PhD/Thesis/Chapter2/Data/WISE/Match_BG_20220705.csv",
    #     max_time_diff=3.0,
    #     window_size=3,
    # )
    #
    # test.to_csv(
    #     path_or_buf="/D/Documents/PhD/Thesis/Chapter2/Data/WISE/Lt_pixex_ACI12A.csv"
    # )
