from reverie import ReveCube
from reverie.converter import wise

if __name__ == "__main__":
    # l1 = wise.read_pix.Pix(
    #     image_dir="/D/Data/TEST/TEST_WISE/ACI-12A/",
    #     image_name="220705_ACI-12A-WI-1x1x1_v01",
    # )
    #
    # l1.to_reve_nc()

    reve_nc = ReveCube.from_reve_nc(
        "/D/Data/TEST/TEST_WISE/ACI-12A/220705_ACI-12A-WI-1x1x1_v01-L1C.nc"
    )

    test = reve_nc.extract_pixel(
        "/D/Documents/PhD/Thesis/Chapter2/Data/WISE/Match_BG_20220705.csv",
        max_time_diff=3.0,
        window_size=3,
    )

    test.to_csv(
        path_or_buf="/D/Documents/PhD/Thesis/Chapter2/Data/WISE/Lt_pixex_ACI12A.csv"
    )
