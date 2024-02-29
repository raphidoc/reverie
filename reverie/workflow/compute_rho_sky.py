import netCDF4 as nc
import scipy as sp
import pandas as pd

rho_sky_nc = nc.Dataset(
    "/D/Documents/PhD/Thesis/Chapter2/Data/ACOLITE-RSKY-202102-82W-MOD2.nc",
    "r",
    format="NETCDF4",
)

insitu_data = pd.read_csv(
    "/D/Documents/PhD/Thesis/Chapter2/Data/WISE/pixex/Lt_rhow_merged.csv"
)

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
        insitu_data["Wavelength"]*1e-3,
        insitu_data["RelativeAzimuth_mean"],
        insitu_data["ViewZen_mean"],
        insitu_data["SolZen_mean"],
        2,
        0.02
    ),
)

insitu_data["rho_sky"] = res
insitu_data.to_csv(
        path_or_buf="/D/Documents/PhD/Thesis/Chapter2/Data/WISE/pixex/Lt_rho_w_rho_sky_merged.csv"
    )

if __name__ == "__name__":
    pass
