import os

from reverie import ReveCube
from reverie.converter import wise


if __name__ == "__main__":
    image_dir = "/D/Data/WISE/"
    matchup_dir = "/D/Documents/PhD/Thesis/Chapter2/Data/WISE/pixex"

    matchups = [
        # "ACI-12A/220705_ACI-12A-WI-1x1x1_v01-L1CG.nc",
        # "MC-50A/190818_MC-50A-WI-2x1x1_v02-L1CG.nc",
        "MC-37A/190818_MC-37A-WI-1x1x1_v02-L1CG.nc",
        # "MC-10A/190820_MC-10A-WI-1x1x1_v02-L1G.nc",
    ]

    for matchup in matchups:
        reve_cube = ReveCube.from_reve_nc(os.path.join(image_dir, matchup))

        test = reve_cube.extract_pixel(
            os.path.join(matchup_dir, "insitu_table.csv"),
            max_time_diff=3.0,
            window_size=7,
        )

        image_name = os.path.basename(reve_cube.in_path).split(".")[0]

        output_name = f"{image_name}_pixex.csv"

        test.to_csv(
            path_or_buf=os.path.join("/D/Documents/PhD/Thesis/Chapter2/Data/WISE/pixex/", output_name)
        )
