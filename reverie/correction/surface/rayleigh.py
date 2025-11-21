from math import cos, sin, pow, exp
import numpy as np

from reverie.correction.surface.water_refractive_index import get_water_refractive_index

# function ray_tau
# computes Rayleigh optical thickness for wl in microns
# QV 2016-12-14

def ray_tau(wl, Patm=1013.25):
    # Wavelength in um
    wl = wl * 1e-3
    tau_ray = Patm/1013.25*(0.008569*np.power(wl,-4)*(1.+0.0113*np.power(wl,-2)+0.00013*np.power(wl,-4)))
    return tau_ray


def fresnel_reflectance(theta, n_w):
    n_air = 1

    theta_t = np.arcsin(n_air / n_w * np.sin(theta))
    r = 0.5 * ((np.power(np.sin(theta - theta_t), 2) / np.power(np.sin(theta + theta_t), 2)) +
                (np.power(np.tan(theta - theta_t), 2) / np.power(np.tan(theta + theta_t), 2)))
    return r

# function ray_phase
# computes Rayleigh phase function for given geometry
# QV 2016-12-14
# MR 2025-11-21

def ray_phase(theta_s, theta_v, phi_s, phi_v, n_w):
    delta_phi = abs(phi_s - phi_v)
    costheta_plus = np.cos(theta_s)*np.cos(theta_v) - np.sin(theta_s)*np.sin(theta_v)*np.cos(delta_phi)
    phase_r_theta = (0.75 * (1. + np.power(costheta_plus, 2.)))

    fr_theta_s = fresnel_reflectance(theta_s, n_w)
    fr_theta_v = fresnel_reflectance(theta_v, n_w)

    phase_r = (fr_theta_s + fr_theta_v) * phase_r_theta

               
    return phase_r


# function ray_tr
# computes Rayleigh transmittance for given geometry
# QV 2016-12-14

def ray_tr(wl, theta_s, theta_v, Patm=1013.25):
    tau_ray = ray_tau(wl, Patm=Patm)
    ray_tr = (1.+exp(-1.*tau_ray/cos(theta_v))) * (1.+exp(-1.*tau_ray/cos(theta_s))) / 4.
    return ray_tr


# function sky_refl
# computes diffuse sky reflectance
# QV 2016-12-14

def sky_refl(theta, n_w=1.34):
    # angle of transmittance theta_t for air incident rays (Mobley, 1994 p156)
    theta_t = np.arcsin( 1. / n_w * np.sin(theta))
    r_int=0.5 * (np.power(np.sin(theta - theta_t) / np.sin(theta + theta_t ), 2) +
              np.power(np.tan(theta - theta_t) / np.tan(theta + theta_t), 2))
    return r_int

# function ray_refl
# computes Rayleigh reflectance for given geometry
# QV 2016-12-14
# MR 2025-11-21

def get_sky_glint(wl, theta_s, theta_v, phi_s, phi_v, Patm):

    wl = np.asarray(wl)
    theta_s = np.asarray(theta_s)
    theta_v = np.asarray(theta_v)
    phi_s = np.asarray(phi_s)
    phi_v = np.asarray(phi_v)

    # theta_s, theta_v, phi_s, phi_v = np.broadcast_arrays(
    #     np.asarray(theta_s), np.asarray(theta_v),
    #     np.asarray(phi_s), np.asarray(phi_v)
    # )

    if (wl.ndim == 1) & (theta_s.ndim == 2):
        wl_ = wl[:, None, None]  # shape (len_wl, 1, 1)
        theta_s_ = theta_s[None, :, :]  # shape (1, y, x)
        theta_v_ = theta_v[None, :, :]
        phi_s_ = phi_s[None, :, :]
        phi_v_ = phi_v[None, :, :]
    else:
        wl_ = wl
        theta_s_ = theta_s
        theta_v_ = theta_v
        phi_s_ = phi_s
        phi_v_ = phi_v

    n_w_ = get_water_refractive_index(
        salinity = 30,
        temperature = 12,
        wavelength = wl_
    )

    tau_ray = ray_tau(wl_, Patm=Patm)
    phase_ray = ray_phase(theta_s_, theta_v_, phi_s_, phi_v_, n_w_)

    sky_glint = (tau_ray * phase_ray) / (4. * np.cos(theta_s) * np.cos(theta_v))

    return sky_glint

# # function ray_phase_nosky
# # computes Rayleigh phase function for given geometry (no diffuse sky reflectance)
# # QV 2016-12-14
#
# def ray_phase_nosky(theta_s,theta_v,phi_s, phi_v):
#     costheta_min = -1. * cos(theta_s)*cos(theta_v) - sin(theta_s)*sin(theta_v)*cos(abs(phi_s-phi_v))
#     phase_r= (0.75*(1.+pow(costheta_min,2.)))
#     return phase_r
#
# # function ray_refl_nosky
# # computes Rayleigh reflectance for given geometry (no diffuse sky reflectance)
# # QV 2016-12-14
#
# def ray_refl_nosky(wl, theta_s, theta_v, phi_s, phi_v, Patm=1013.25, tau_ray=None):
#     if tau_ray is None: tau_ray = ray_tau(wl, Patm=Patm)
#     phase_ray = ray_phase_nosky(theta_s,theta_v,phi_s, phi_v)
#     rho_ray = (tau_ray * phase_ray) / (4. * cos(theta_s)*cos(theta_v))
#     return rho_ray
#
#
# # function ray_phase_onlysky
# # computes Rayleigh phase function for given geometry - only diffuse sky reflectance
# # QV 2017-11-14
#
# def ray_phase_onlysky(theta_s,theta_v,phi_s, phi_v):
#     costheta_plus = 1. * cos(theta_s)*cos(theta_v) - sin(theta_s)*sin(theta_v)*cos(abs(phi_s-phi_v))
#     phase_r= (sky_refl(theta_s)+sky_refl(theta_v)) * (0.75*(1.+pow(costheta_plus,2.)))
#     return phase_r
#
# # function ray_refl_onlysky
# # computes Rayleigh reflectance for given geometry - only diffuse sky reflectance
# # QV 2017-11-14
#
# def ray_refl_onlysky(wl, theta_s, theta_v, phi_s, phi_v, Patm=1013.25, tau_ray=None):
#     if tau_ray is None: tau_ray = ray_tau(wl, Patm=Patm)
#     phase_ray = ray_phase_onlysky(theta_s,theta_v,phi_s, phi_v)
#     rho_ray = (tau_ray * phase_ray) / (4. * cos(theta_s)*cos(theta_v))
#     return rho_ray
