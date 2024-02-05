import subprocess
import pandas as pd


def run_viccal_6s(params: str):
    exec_6sv2 = "/home/raphael/PycharmProjects/reverie/reverie/6S/6sV2.1/sixsV2.1"
    process = subprocess.run(exec_6sv2, input=params, text=True, capture_output=True)

    # Check if the process completed successfully
    if process.returncode != 0:
        raise Exception(
            "6S execution failed with error code %s" % process.returncode,
            process.stderr,
        )
    else:
        return process.stdout


if __name__ == "__main__":
    input_data = pd.read_csv(
        "/D/Documents/PhD/Thesis/Chapter2/Data/WISE/pixex/ACI12_merged_data.csv"
    )

    # Initialize the output columns
    input_data["atmospheric_path_radiance"] = 0
    input_data["background_radiance"] = 0
    input_data["pixel_radiance"] = 0

    for i in range(len(input_data)):
        kwargs = {
            "sol_zen": input_data["SolZen.x"][i],
            "sol_azi": input_data["SolAzm.x"][i],
            "view_zen": input_data["ViewZen"][i],
            "view_azi": input_data["ViewAzm"][i],
            "aot550": 0.04,
            "sensor_altitude": -3.049,
            "wavelength": input_data["Wavelength"][i] * 1e-3,
            "water_reflectance": input_data["rhow"][i],
        }

        input_str = "\n".join(
            [
                # Geometrical conditions
                "0 # Geometrical conditions igeom",
                "%(sol_zen).2f %(sol_azi).2f %(view_zen).2f %(view_azi).2f 1 1"
                % kwargs,
                # Atmospheric conditions, gas
                "2 # Atmospheric conditions, gas (idatm)"
                # Aerosol type and concentration
                "2 # Maritime aerosol model",
                "0 # Visibility (km)",
                "%(aot550).2f" % kwargs,
                # Target altitude as pressure in mb
                "0 # target altitude (xps)",
                # Sensor altitude in - km relative to target altitude
                "%(sensor_altitude).4f # Sensor altitude (xpp)" % kwargs,
                # If aircraft simulation (> -110 < 0) enter water vapor, ozone content
                # Aircraft water vapor, ozone and aot550
                "-1.0 -1.0",
                "-1.0",
                # Spectrum selection for simulation
                "-1 # Spectrum monochromatic selection (iwave)",
                "%(wavelength).3f" % kwargs,
                # Ground reflectance
                "0 # Ground reflectance (inhomo)",
                "0",
                "0",
                "%(water_reflectance).6f" % kwargs,
                # Atmospheric correction
                "-1 # Atmospheric correction (irapp)",
            ]
        )

        # with open("/home/raphael/PycharmProjects/reverie/reverie/6S/in.txt", "w") as f:
        #     f.writelines(input_str)

        res = run_viccal_6s(input_str)

        import json

        test = json.loads(res)

        input_data[list(test.keys())[0]][i] = test[list(test.keys())[0]]
        input_data[list(test.keys())[1]][i] = test[list(test.keys())[1]]
        input_data[list(test.keys())[2]][i] = test[list(test.keys())[2]]

    input_data.to_csv(
        "/D/Documents/PhD/Thesis/Chapter2/Data/WISE/6S/6S_ACI12_merged_data.csv"
    )

    # with open("/home/raphael/PycharmProjects/reverie/reverie/6S/out.txt", "w") as f:
    #     f.writelines(res)
