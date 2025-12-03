import os

from reverie import ReveCube
from reverie.converter import wise


if __name__ == "__main__":
    image_dir = "/D/Data/WISE/"
    matchup_dir = "/D/Documents/phd/thesis/3_chapter/data/wise/pixex"

    matchups = [
        "ACI-12A/220705_ACI-12A-WI-1x1x1_v01-l2rg.nc",
        "ACI-13A/220705_ACI-13A-WI-1x1x1_v01-l2rg.nc",
    ]

    for matchup in matchups:
        reve_cube = ReveCube.from_reve_nc(os.path.join(image_dir, matchup))

        test = reve_cube.extract_pixel(
            os.path.join(matchup_dir, "sas_coord.csv"),
            max_time_diff=5,
            window_size=7,
        )

        image_name = os.path.basename(reve_cube.in_path).split(".")[0]

        output_name = f"{image_name}_pixex.csv"

        test[0].to_csv(
            path_or_buf=os.path.join(matchup_dir, output_name)
        )
