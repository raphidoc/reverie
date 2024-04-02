import netCDF4 as nc
import xarray as xr
import scipy as sp
import pandas as pd

#pd.options.mode.copy_on_write = True

rho_sky_nc = nc.Dataset(
    "/D/Documents/PhD/Thesis/Chapter2/Data/ACOLITE-RSKY-202102-82W-MOD2.nc",
    "r",
    format="NETCDF4",
)

insitu_data = pd.read_csv(
    "/D/Documents/PhD/Thesis/Chapter2/Data/WISE/pixex/radiance_at_sensor_rho_w_merged.csv"
)

# Convert relative azimuth to the 0 - 180 range
for ix in range(0, len(insitu_data["relative_azimuth_mean"])):
    if 0 <= insitu_data["relative_azimuth_mean"][ix] <= 180:
        continue
    elif 180 < insitu_data["relative_azimuth_mean"][ix] <= 360:
        insitu_data.loc[ix, ("relative_azimuth_mean")] = (
            360 - insitu_data.loc[ix, ("relative_azimuth_mean")]
        )
    else:
        raise ValueError("relative azimuth is not in a valid 0 - 360 range")

rho_sky_nc.dimensions
rho_sky_nc.variables["lut"]

res = sp.interpolate.interpn(
    points=(
        rho_sky_nc.wave[()],
        rho_sky_nc.azi[()],
        rho_sky_nc.thv[()],
        rho_sky_nc.ths[()],
        rho_sky_nc.wind[()],
        rho_sky_nc.tau[()],
    ),
    values=rho_sky_nc.variables["lut"][:, :, :, :, :, :],
    xi=(
        insitu_data["Wavelength"] * 1e-3,
        insitu_data["relative_azimuth_mean"],
        insitu_data["view_zenith_mean"],
        insitu_data["sun_zenith_mean"],
        2,
        0.12,
    ),
)

insitu_data["rho_sky"] = res
insitu_data.to_csv(
    path_or_buf="/D/Documents/PhD/Thesis/Chapter2/Data/WISE/pixex/radiance_at_sensor_rho_w_rho_sky_merged.csv"
)

if __name__ == "__name__":
    pass
