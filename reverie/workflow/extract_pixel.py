from reverie import ReveCube
from reverie.converter import wise

if __name__ == "__main__":

    # ACI-12
    # l1 = wise.read_pix.Pix(
    #     pix_dir="/D/Data/TEST/TEST_WISE/ACI-12A/220705_ACI-12A-WI-1x1x1_v01-L1CG.dpix",
    # )
    #
    # l1.to_reve_nc()
    #
    # reve_nc = ReveCube.from_reve_nc(
    #     "/D/Data/TEST/TEST_WISE/ACI-12A/220705_ACI-12A-WI-1x1x1_v01-L1C.nc"
    # )
    #
    # test = reve_nc.extract_pixel(
    #     "/D/Documents/PhD/Thesis/Chapter2/Data/WISE/Match_BG_20220705.csv",
    #     max_time_diff=3.0,
    #     window_size=3,
    # )
    #
    # test.to_csv(
    #     path_or_buf="/D/Documents/PhD/Thesis/Chapter2/Data/WISE/pixex/Lt_pixex_ACI-12A.csv"
    # )

    # MC-50
    l1 = wise.read_pix.Pix(
        pix_dir="/D/Data/WISE/MC-50A/190818_MC-50A-WI-2x1x1_v02-L1CG.dpix",
    )

    # l1.to_reve_nc()
    #
    # reve_nc = ReveCube.from_reve_nc(
    #     "/D/Data/TEST/TEST_WISE/MC-50A/190818_MC-50A-WI-2x1x1_v02-L1CG.nc"
    # )
    #
    # test = reve_nc.extract_pixel(
    #     "/D/Documents/PhD/Thesis/Chapter2/Data/WISE/Match_MP_20190818.csv",
    #     max_time_diff=3.0,
    #     window_size=3,
    # )
    #
    # test.to_csv(
    #     path_or_buf="/D/Documents/PhD/Thesis/Chapter2/Data/WISE/pixex/Lt_pixex_MC-50A.csv"
    # )

    # MC-10
    # l1 = wise.read_pix.Pix(
    #     pix_dir="/D/Data/TEST/TEST_WISE/MC-10A/",
    # )
    #
    # l1.to_reve_nc()
    #
    # reve_nc = ReveCube.from_reve_nc(
    #     "/D/Data/TEST/TEST_WISE/MC-10A/190820_MC-10A-WI-1x1x1_v02-L1G.nc"
    # )
    #
    # test = reve_nc.extract_pixel(
    #     "/D/Documents/PhD/Thesis/Chapter2/Data/WISE/Match_MP_20190820.csv",
    #     max_time_diff=5,
    #     window_size=3,
    # )
    #
    # test.to_csv(
    #     path_or_buf="/D/Documents/PhD/Thesis/Chapter2/Data/WISE/pixex/Lt_pixex_MC-10A.csv"
    # )