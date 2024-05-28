from reverie import ReveCube

reve = ReveCube.from_reve_nc("/D/Data/WISE/ACI-12A/220705_ACI-12A-WI-1x1x1_v01-L1CG.nc")

# Do gas correction with 6S output "global_gas_trans_total"
# It integrates all gas transmittance in the downward and upward direction
# Dimensions of the LUT are {sun_zenith, view_zenith, relative_azimuth, water_vapor, ozone, target_pressure, sensor_altitude}