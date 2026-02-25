# #!/usr/bin/env python3
# from __future__ import annotations
#
# import os
# import json
# import math
# import logging
# import sqlite3
# import tempfile
# import multiprocessing as mp
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from typing import Any, Dict, Optional, Sequence, Tuple, List
# import functools
#
#
# import numpy as np
# import pandas as pd
# import geopandas as gpd
#
#
# # =============================================================================
# # Logging / silence
# # =============================================================================
# _STAN_MODEL = None
# _BASE: dict[str, Any] | None = None
#
#
# def silence_everything() -> None:
#     logging.disable(logging.ERROR)
#     for name in ("cmdstanpy", "cmdstanpy.model", "cmdstanpy.utils", "cmdstanpy.stanfit"):
#         lg = logging.getLogger(name)
#         lg.setLevel(logging.CRITICAL)
#         lg.propagate = False
#     os.environ["CMDSTANPY_LOG_LEVEL"] = "CRITICAL"
#     os.environ["CMDSTANPY_SILENT"] = "true"
#
#
# # =============================================================================
# # Small numerics helpers
# # =============================================================================
# def interp1_flat(x: np.ndarray, y: np.ndarray, xout: np.ndarray) -> np.ndarray:
#     """Linear interpolation with flat extrapolation (like R approx(rule=2))."""
#     x = np.asarray(x, float)
#     y = np.asarray(y, float)
#     xout = np.asarray(xout, float)
#
#     order = np.argsort(x)
#     x = x[order]
#     y = y[order]
#
#     yout = np.interp(xout, x, y)
#     yout = np.where(xout < x[0], y[0], yout)
#     yout = np.where(xout > x[-1], y[-1], yout)
#     return yout
#
#
# def pca_basis(Xc: np.ndarray) -> np.ndarray:
#     """
#     PCA basis via SVD. Xc must be centered: [n_spec, n_wl].
#     Returns V (loadings): [n_wl, n_comp], columns are PCs.
#     """
#     _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
#     return Vt.T
#
#
# # =============================================================================
# # Optics LUTs (a_w, a0, a1) from CSV + bb_w formula
# # =============================================================================
# def load_optics_luts_from_csv(
#     wavelength: Sequence[float],
#     a_w_csv: str,
#     a0_a1_phyto_csv: str,
# ) -> Dict[str, np.ndarray]:
#     wl = np.asarray(wavelength, float)
#     if wl.ndim != 1 or wl.size < 2:
#         raise ValueError("wavelength must be 1D length>=2")
#
#     a_w_df = pd.read_csv(a_w_csv)
#     a01_df = pd.read_csv(a0_a1_phyto_csv)
#
#     for c in ("wavelength", "a_w"):
#         if c not in a_w_df.columns:
#             raise ValueError(f"{a_w_csv} missing column '{c}'")
#     for c in ("wavelength", "a0", "a1"):
#         if c not in a01_df.columns:
#             raise ValueError(f"{a0_a1_phyto_csv} missing column '{c}'")
#
#     a_w = interp1_flat(a_w_df["wavelength"].to_numpy(float), a_w_df["a_w"].to_numpy(float), wl)
#     a0 = interp1_flat(a01_df["wavelength"].to_numpy(float), a01_df["a0"].to_numpy(float), wl)
#     a1 = interp1_flat(a01_df["wavelength"].to_numpy(float), a01_df["a1"].to_numpy(float), wl)
#
#     # Zhang & Hu-ish pure-water bb_w parameterization you used before
#     bb_w = 0.00111 * (wl / 500.0) ** (-4.32)
#
#     return {"a_w": a_w, "a0": a0, "a1": a1, "bb_w": bb_w}
#
#
# # =============================================================================
# # Low-rank PCA prior from replicate bottom spectra (your R logic, ported)
# # =============================================================================
# def build_lowrank_pca_prior_from_replicates(
#     wavelength: Sequence[float],                 # Stan grid [n_wl]
#     rb_obs_path: str,
#     q: int = 8,
#     tau_floor: float = 1e-6,
#     sigma2_floor: float = 1e-8,
#     class_col: str = "class",
#     wl_col: str = "wavelength",
#     rb_col: str = "r_b",
#     rep_id_col_candidates: Tuple[str, ...] = ("time_target",),
# ) -> Dict[str, Any]:
#     wl_target = np.asarray(wavelength, float)
#     n_wl = wl_target.size
#
#     rb_obs = pd.read_csv(rb_obs_path)
#     for c in (class_col, wl_col, rb_col):
#         if c not in rb_obs.columns:
#             raise ValueError(f"{rb_obs_path} missing column '{c}'")
#
#     rb_obs[class_col] = rb_obs[class_col].astype(str)
#     rb_obs[wl_col] = pd.to_numeric(rb_obs[wl_col], errors="coerce")
#     rb_obs[rb_col] = pd.to_numeric(rb_obs[rb_col], errors="coerce")
#
#     rep_id_col = next((c for c in rep_id_col_candidates if c in rb_obs.columns), None)
#     if rep_id_col is None:
#         rb_obs = rb_obs.sort_values([class_col, wl_col], kind="mergesort")
#
#         def _add_rep_id(df: pd.DataFrame) -> pd.DataFrame:
#             w = df[wl_col].to_numpy(float)
#             rep = np.ones(len(df), dtype=int)
#             rep[1:] = (np.diff(w) < 0).astype(int)
#             df = df.copy()
#             df["rep_id"] = np.cumsum(rep)
#             return df
#
#         rb_obs = rb_obs.groupby(class_col, group_keys=False).apply(_add_rep_id)
#         rep_id_col = "rep_id"
#
#     spectra_df = (
#         rb_obs[[class_col, rep_id_col, wl_col, rb_col]]
#         .pivot_table(index=[class_col, rep_id_col], columns=wl_col, values=rb_col, aggfunc="first")
#         .reset_index()
#     )
#
#     wl_native = np.array(
#         [c for c in spectra_df.columns if isinstance(c, (int, float, np.integer, np.floating))],
#         float,
#     )
#     if wl_native.size < 2:
#         raise ValueError("Not enough wavelength columns found after pivot.")
#
#     wl_native = np.sort(wl_native)
#     X_native = spectra_df[wl_native].to_numpy(float)  # (n_spec, n_wl_native)
#
#     X = np.full((X_native.shape[0], n_wl), np.nan, float)
#     for i in range(X_native.shape[0]):
#         y = X_native[i, :]
#         ok = np.isfinite(y)
#         if ok.sum() < 2:
#             continue
#         X[i, :] = interp1_flat(wl_native[ok], y[ok], wl_target)
#
#     if not np.isfinite(X).all():
#         col_med = np.nanmedian(X, axis=0)
#         bad = ~np.isfinite(X)
#         X[bad] = np.take(col_med, np.where(bad)[1])
#
#     cls_vec = spectra_df[class_col].astype(str).to_numpy()
#     classes = sorted(pd.unique(cls_vec).tolist())
#     K = len(classes)
#
#     mu_global = X.mean(axis=0)
#     Xc = X - mu_global[None, :]
#     V = pca_basis(Xc)  # (n_wl, n_comp)
#     q_eff = int(min(q, V.shape[1], n_wl))
#     rb_U = V[:, :q_eff].copy()  # (n_wl, q)
#
#     rb_mu_k: List[np.ndarray] = []
#     rb_tau_k: List[np.ndarray] = []
#     rb_sigma2_k = np.zeros(K, float)
#     counts = np.zeros(K, int)
#
#     for i, cl in enumerate(classes):
#         idx = np.where(cls_vec == cl)[0]
#         counts[i] = idx.size
#
#         Xk = X[idx, :]
#         mu_k = Xk.mean(axis=0)
#         rb_mu_k.append(mu_k.astype(float, copy=True))
#
#         Rk = Xk - mu_k[None, :]
#         Ak = Rk @ rb_U  # (n_k, q)
#
#         tau = Ak.std(axis=0, ddof=1) if Ak.shape[0] > 1 else np.zeros(q_eff, float)
#         tau[~np.isfinite(tau)] = 0.0
#         tau = np.maximum(tau, float(tau_floor))
#         rb_tau_k.append(tau.astype(float, copy=True))
#
#         Rhat = Ak @ rb_U.T
#         Ek = Rk - Rhat
#         sigma2 = float(np.mean(Ek**2))
#         rb_sigma2_k[i] = max(sigma2, float(sigma2_floor))
#
#     rb_pi = counts.astype(float) / float(counts.sum()) if counts.sum() else np.full(K, 1.0 / K)
#
#     # CmdStan JSON/data format: rb_U as list-of-rows; arrays as lists
#     return {
#         "K": int(K),
#         "q": int(q_eff),
#         "rb_U": rb_U.tolist(),                              # [n_wl][q]
#         "rb_mu_k": [v.tolist() for v in rb_mu_k],           # list K of [n_wl]
#         "rb_tau_k": [v.tolist() for v in rb_tau_k],         # list K of [q]
#         "rb_sigma2_k": rb_sigma2_k.tolist(),                # [K]
#         "rb_pi": rb_pi.tolist(),                            # [K], sums to 1
#         "classes": classes,
#     }
#
#
# # =============================================================================
# # Compile Stan once (parent) and load exe in workers
# # =============================================================================
# def compile_model(stan_model_path: str, user_header: str | None = None) -> str:
#     from cmdstanpy import CmdStanModel
#     m = CmdStanModel(stan_file=stan_model_path, user_header=user_header)
#     return m.exe_file
#
#
# # =============================================================================
# # Worker init: build _BASE for the *new* low-rank PCA prior model
# # =============================================================================
# def _init_worker(
#     exe_file: str,
#     wavelength: np.ndarray,
#     *,
#     water_type: int,
#     shallow: int,
#     # LUT paths
#     a_w_csv: str,
#     a0_a1_phyto_csv: str,
#     # replicate spectra path
#     rb_obs_path: str,
#     # PCA/prior config
#     q: int,
#     tau_floor: float,
#     sigma2_floor: float,
#     # prior hyperparameters (must match Stan data names)
#     priors: dict,
# ) -> None:
#     global _STAN_MODEL, _BASE
#     silence_everything()
#     from cmdstanpy import CmdStanModel
#
#     _STAN_MODEL = CmdStanModel(exe_file=exe_file)
#
#     wl = np.asarray(wavelength, float)
#     optics = load_optics_luts_from_csv(wl, a_w_csv=a_w_csv, a0_a1_phyto_csv=a0_a1_phyto_csv)
#     pca = build_lowrank_pca_prior_from_replicates(
#         wavelength=wl,
#         rb_obs_path=rb_obs_path,
#         q=q,
#         tau_floor=tau_floor,
#         sigma2_floor=sigma2_floor,
#     )
#
#     base = {
#         "n_wl": int(wl.size),
#         "wavelength": wl.tolist(),
#         "a_w": optics["a_w"].tolist(),
#         "a0": optics["a0"].tolist(),
#         "a1": optics["a1"].tolist(),
#         "bb_w": optics["bb_w"].tolist(),
#         "water_type": int(water_type),
#         "theta_sun": 30.0,     # placeholder (set per segment)
#         "theta_view": 0.0,     # placeholder (set per segment)
#         "shallow": int(shallow),
#         "K": pca["K"],
#         "q": pca["q"],
#         "rb_U": pca["rb_U"],
#         "rb_mu_k": pca["rb_mu_k"],
#         "rb_tau_k": pca["rb_tau_k"],
#         "rb_sigma2_k": pca["rb_sigma2_k"],
#         "rb_pi": pca["rb_pi"],
#     }
#     base.update(priors)
#     _BASE = base
#
#
# # =============================================================================
# # Segment inversion (new model): use Stan’s generated quantities r_b_hat, rrs_hat
# # =============================================================================
# def invert_one_segment(job: dict, stan_cfg: dict, out_vars: tuple[str, ...]) -> dict:
#     global _STAN_MODEL, _BASE
#     assert _STAN_MODEL is not None
#     assert _BASE is not None
#
#     seg_id = int(job["seg_id"])
#     wl = job["wl_nm"].astype(float)
#
#     rho_est = job["rho_med"].astype(float)
#     rho_unc = job["rho_mad"].astype(float)
#
#     # match your pixel conversion
#     rrs_0p = rho_est / np.pi
#     rrs_0m = rrs_0p / (0.52 + 1.7 * rrs_0p)
#     if not np.all(np.isfinite(rrs_0m)):
#         return {"seg_id": seg_id, "ok": False, "err": "non-finite rrs_0m"}
#
#     rrs_0p_unc = rho_unc / np.pi
#     rrs_0m_unc = rrs_0p_unc / (0.52 + 1.7 * rrs_0p_unc)
#     rrs_0m_unc = np.maximum(rrs_0m_unc, 1e-8)
#
#     stan_data = dict(_BASE)
#     stan_data["rrs_obs"] = rrs_0m.tolist()
#     stan_data["sigma_rrs"] = rrs_0m_unc.tolist()
#     stan_data["theta_sun"] = float(job["theta_sun"])
#     stan_data["theta_view"] = float(job["theta_view"])
#
#     # new model uses depth prior in data (not fixed h_w)
#     stan_data["h_w_mu"] = float(job["h_w_mu"])
#     stan_data["h_w_sd"] = float(job["h_w_sd"])
#     # keep h_w_sd from _BASE (or tighten per segment if you want)
#
#     pid_dir = tempfile.mkdtemp(prefix="cmdstan_", dir="/tmp")
#
#     try:
#         if stan_cfg["method"] == "optimize":
#             fit = _STAN_MODEL.optimize(
#                 data=stan_data,
#                 seed=stan_cfg["seed"],
#                 output_dir=pid_dir,
#                 show_console=False,
#             )
#             est = fit.optimized_params_dict
#         else:
#             fit = _STAN_MODEL.sample(
#                 data=stan_data,
#                 seed=stan_cfg["seed"],
#                 chains=int(stan_cfg["chains"]),
#                 parallel_chains=1,
#                 iter_warmup=int(stan_cfg["iter_warmup"]),
#                 iter_sampling=int(stan_cfg["iter_sampling"]),
#                 show_progress=False,
#                 show_console=False,
#                 output_dir=pid_dir,
#             )
#             draws = fit.draws_pd()
#             est = {v: float(draws[v].mean()) for v in draws.columns}
#
#         scalars = {v: float(est[v]) for v in out_vars if v in est}
#
#         # Pull r_b_hat and rrs_hat from generated quantities
#         nwl = len(wl)
#         r_b_hat = np.full(nwl, np.nan, dtype=float)
#         rrs_hat = np.full(nwl, np.nan, dtype=float)
#
#         for i in range(nwl):
#             kb = f"r_b_hat[{i + 1}]"
#             kr = f"rrs_hat[{i + 1}]"
#             if kb in est:
#                 r_b_hat[i] = float(est[kb])
#             if kr in est:
#                 rrs_hat[i] = float(est[kr])
#
#         # Convert modeled Rrs (below-surface) back to rho_w
#         # Inverse of: rrs_0m = rrs_0p / (0.52 + 1.7*rrs_0p)
#         # => rrs_0p = (0.52 * rrs_0m) / (1 - 1.7 * rrs_0m)
#         rho_hat = np.full(nwl, np.nan, dtype=float)
#         ok = np.isfinite(rrs_hat) & (1 - 1.7 * rrs_hat > 1e-12)
#         rrs_0p_hat = np.full(nwl, np.nan, dtype=float)
#         rrs_0p_hat[ok] = (0.52 * rrs_hat[ok]) / (1 - 1.7 * rrs_hat[ok])
#         rho_hat[ok] = rrs_0p_hat[ok] * np.pi
#
#         r_b_hat = np.clip(np.nan_to_num(r_b_hat, nan=0.0), 0.0, 1.0)
#
#         return {
#             "seg_id": seg_id,
#             "ok": True,
#             "scalars": scalars,
#             "wl_nm": wl.astype(float),
#             "r_b_med": r_b_hat.astype(float),
#             "rho_hat": rho_hat.astype(float),
#         }
#
#     except Exception as e:
#         return {"seg_id": seg_id, "ok": False, "err": repr(e)}
#
#
# # =============================================================================
# # SQLite helpers (write non-spatial tables into GPKG)
# # =============================================================================
# def sql_safe(name: str) -> str:
#     return (
#         name.replace("[", "_")
#         .replace("]", "")
#         .replace(".", "_")
#         .replace("-", "_")
#         .replace(" ", "_")
#         .replace("(", "_")
#         .replace(")", "_")
#     )
#
#
# def write_table_sqlite(gpkg_path: str, table: str, df: pd.DataFrame, overwrite: bool = True) -> None:
#     conn = sqlite3.connect(gpkg_path)
#     cur = conn.cursor()
#     if overwrite:
#         cur.execute(f'DROP TABLE IF EXISTS "{table}"')
#
#     cols_def = []
#     for c in df.columns:
#         if np.issubdtype(df[c].dtype, np.integer):
#             t = "INTEGER"
#         elif np.issubdtype(df[c].dtype, np.floating):
#             t = "REAL"
#         else:
#             t = "TEXT"
#         cols_def.append(f'"{c}" {t}')
#
#     cur.execute(f'CREATE TABLE "{table}" ({", ".join(cols_def)})')
#
#     placeholders = ",".join(["?"] * len(df.columns))
#     cols = ", ".join(f'"{c}"' for c in df.columns)
#     ins = f'INSERT INTO "{table}" ({cols}) VALUES ({placeholders})'
#     cur.executemany(ins, df.itertuples(index=False, name=None))
#
#     conn.commit()
#     conn.close()
#
#
# # =============================================================================
# # Main driver: read segments.gpkg -> invert -> write segments_inverted.gpkg
# # =============================================================================
# def run_gpkg_inversion_to_new_file(
#     in_gpkg: str,
#     out_gpkg: str,
#     segments_layer: str = "segments",
#     spectra_table: str = "spectra",
#     aux_table: str = "aux",
#     seg_id_field: str = "seg_id",
#     wl_field: str = "wl_nm",
#     rho_med_field: str = "rho_med",
#     rho_mad_field: str = "rho_mad",
#     sun_field: str = "sun_zenith_med",
#     view_field: str = "view_zenith_med",
#     hw_field: str = "h_w_mu",
#     stan_model_path: str = "",
#     out_vars: tuple[str, ...] = ("chl", "a_g_440", "spm", "h_w", "sigma_model", "lp__"),
#     method: str = "optimize",
#     n_workers: int | None = None,
#     h_w_mu_override: float | None = None,
#     h_w_sd_override: float | None = None,
# ) -> int:
#     # ---- inputs
#     gdf = gpd.read_file(in_gpkg, layer=segments_layer)
#
#     conn = sqlite3.connect(in_gpkg)
#     spec = pd.read_sql_query(
#         f"""
#         SELECT
#             {seg_id_field}     AS seg_id,
#             {wl_field}         AS wl_nm,
#             {rho_med_field}    AS rho_med,
#             {rho_mad_field}    AS rho_mad
#         FROM "{spectra_table}"
#         """,
#         conn,
#     )
#     aux = pd.read_sql_query(
#         f"""
#         SELECT
#             {seg_id_field}   AS seg_id,
#             {sun_field}      AS theta_sun,
#             {view_field}     AS theta_view,
#             {hw_field}       AS h_w
#         FROM "{aux_table}"
#         """,
#         conn,
#     )
#     conn.close()
#
#     merged = spec.merge(aux, on="seg_id", how="inner")
#
#     jobs: list[dict] = []
#     for seg_id, gg in merged.groupby("seg_id"):
#         gg = gg.sort_values("wl_nm")
#         if not np.isfinite(gg["theta_sun"].iloc[0]):
#             continue
#
#         # NEW: choose h_w_mu source
#         if h_w_mu_override is None:
#             h_w_mu = float(gg["h_w_mu"].iloc[0])
#             if not np.isfinite(h_w_mu):
#                 continue
#         else:
#             h_w_mu = float(h_w_mu_override)
#
#         if h_w_sd_override is None:
#             h_w_sd = float(gg["h_w_sd"].iloc[0])
#             if not np.isfinite(h_w_sd):
#                 continue
#         else:
#             h_w_sd = float(h_w_sd_override)
#
#         jobs.append(
#             {
#                 "seg_id": int(seg_id),
#                 "wl_nm": gg["wl_nm"].to_numpy(float),
#                 "rho_med": gg["rho_med"].to_numpy(float),
#                 "rho_mad": gg["rho_mad"].to_numpy(float),
#                 "theta_sun": float(gg["theta_sun"].iloc[0]),
#                 "theta_view": float(gg["theta_view"].iloc[0]),
#                 "h_w_mu": h_w_mu,
#                 "h_w_sd": h_w_sd,
#             }
#         )
#
#     if not jobs:
#         raise RuntimeError("No valid segments to invert (check aux join / NaNs).")
#
#     # ---- multiprocessing config
#     if n_workers is None:
#         n_workers = max(1, (os.cpu_count() or 2) - 1)
#
#     wavelength = np.asarray(jobs[0]["wl_nm"], dtype=float)
#
#     # Stan compile once
#     user_header = "/home/raphael/R/SABER/inst/stan/fct_rtm_stan.hpp"
#     exe_file = compile_model(stan_model_path, user_header=user_header)
#
#     # Paths for LUTs + replicate spectra (PCA prior)
#     a_w_csv = "/home/raphael/R/SABER/inst/extdata/a_w.csv"
#     a0_a1_phyto_csv = "/home/raphael/R/SABER/inst/extdata/a0_a1_phyto.csv"
#     rb_obs_path = "/home/raphael/R/SABER/inst/extdata/r_b_gamache_obs.csv"
#
#     # Prior hypers required by Stan data block
#     priors = dict(
#         a_nap_star_mu=0.0051, a_nap_star_sd=0.0012,
#         bb_p_star_mu=0.0047,  bb_p_star_sd=0.0012,
#         a_g_s_mu=0.017,       a_g_s_sd=0.0012,
#         a_nap_s_mu=0.006,     a_nap_s_sd=0.0012,
#         bb_p_gamma_mu=0.65,   bb_p_gamma_sd=0.12,
#         h_w_mu=3.0,           h_w_sd=2.0,    # overwritten per-segment with bathy median
#     )
#
#     initargs = (
#         exe_file,
#         wavelength,
#     )
#     initkwargs = dict(
#         water_type=2,
#         shallow=1,
#         a_w_csv=a_w_csv,
#         a0_a1_phyto_csv=a0_a1_phyto_csv,
#         rb_obs_path=rb_obs_path,
#         q=8,
#         tau_floor=1e-6,
#         sigma2_floor=1e-8,
#         priors=priors,
#     )
#
#     ctx = mp.get_context("spawn")
#     stan_cfg = dict(method=method, seed=1234, chains=2, iter_warmup=300, iter_sampling=300)
#
#     init_fn = functools.partial(_init_worker, **initkwargs)
#
#     results: list[dict] = []
#     failed: list[dict] = []
#
#     with ProcessPoolExecutor(
#             max_workers=n_workers,
#             mp_context=ctx,
#             initializer=init_fn,
#             initargs=(exe_file, wavelength),
#     ) as ex:
#         futs = [ex.submit(invert_one_segment, job, stan_cfg, out_vars) for job in jobs]
#         for fut in as_completed(futs):
#             r = fut.result()
#             if r.get("ok"):
#                 results.append(r)
#             else:
#                 failed.append(r)
#
#     if not results:
#         err = failed[0].get("err") if failed else "unknown"
#         raise RuntimeError(f"All inversions failed. Example error: {err}")
#
#     # ---- outputs
#     scal_df = pd.DataFrame(
#         [
#             {"seg_id": r["seg_id"], **{sql_safe(k): r["scalars"].get(k, np.nan) for k in out_vars}}
#             for r in results
#         ]
#     )
#
#     gdf[seg_id_field] = gdf[seg_id_field].astype(int)
#     scal_df["seg_id"] = scal_df["seg_id"].astype(int)
#     gdf_out = gdf.merge(scal_df, left_on=seg_id_field, right_on="seg_id", how="left", suffixes=("", "_inv"))
#     if "seg_id_inv" in gdf_out.columns:
#         gdf_out = gdf_out.drop(columns=["seg_id_inv"])
#
#     rb_rows = []
#     for r in results:
#         seg_id = int(r["seg_id"])
#         wl_nm = r["wl_nm"]
#         rb_med = r["r_b_med"]
#         rho_hat = r["rho_hat"]
#         if not (len(wl_nm) == len(rb_med) == len(rho_hat)):
#             raise ValueError(f"Length mismatch seg_id={seg_id}")
#
#         for wl, rb, rho in zip(wl_nm, rb_med, rho_hat):
#             rb_rows.append(
#                 {"seg_id": seg_id, "wl_nm": float(wl), "r_b_hat": float(rb), "rho_hat": float(rho)}
#             )
#     rb_df = pd.DataFrame(rb_rows)
#
#     out_dir = os.path.dirname(out_gpkg)
#     if out_dir:
#         os.makedirs(out_dir, exist_ok=True)
#     if os.path.exists(out_gpkg):
#         os.remove(out_gpkg)
#
#     gdf_out.to_file(out_gpkg, layer=segments_layer, driver="GPKG")
#     write_table_sqlite(out_gpkg, "spectra", rb_df, overwrite=True)
#
#     return len(results)
#
#
# # =============================================================================
# # Run
# # =============================================================================
# if __name__ == "__main__":
#     n_ok = run_gpkg_inversion_to_new_file(
#         in_gpkg="/D/Data/WISE/ACI-12A/el_sma/segmentation_outputs/segments.gpkg",
#         out_gpkg="/D/Data/WISE/ACI-12A/el_sma/segmentation_outputs/segments_inverted.gpkg",
#         segments_layer="segments",
#         spectra_table="spectra",
#         aux_table="aux",
#         stan_model_path="/home/raphael/R/SABER/inst/stan/model_gmm.stan",
#         # include what you actually want saved on segments:
#         out_vars=("chl", "a_g_440", "spm", "h_w", "sigma_model", "lp__"),
#         method="optimize",
#         n_workers=None,
#         # h_w_mu_override= None,
#         # h_w_sd_override= None,
#         h_w_mu_override=5,
#         h_w_sd_override=5,
#     )
#     print("segments inverted:", n_ok)
#
# # from qgis.core import QgsProject, QgsVectorLayer
# #
# # seg_id = [% "seg_id" %]
# #
# # proj = QgsProject.instance()
# #
# # tbl = proj.mapLayersByName("segments_inverted — spectra")[0]  # <-- exact layer name
# # tbl.removeSelection()
# #
# # # build expression depending on seg_id type
# # if isinstance(seg_id, (int, float)):
# #     expr = f"\"seg_id\" = {int(seg_id)}"
# # else:
# #     expr = f"\"seg_id\" = '{seg_id}'"
# #
# # tbl.selectByExpression(expr, QgsVectorLayer.SetSelection)

# #!/usr/bin/env python3
# from __future__ import annotations
#
# import os
# import logging
# import sqlite3
# import tempfile
# import multiprocessing as mp
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from typing import Any, Dict, Optional, Sequence, Tuple
#
# import functools
# import numpy as np
# import pandas as pd
# import geopandas as gpd
#
#
# # =============================================================================
# # Logging / silence
# # =============================================================================
# _STAN_MODEL = None
# _BASE: dict[str, Any] | None = None
#
#
# def silence_everything() -> None:
#     logging.disable(logging.ERROR)
#     for name in ("cmdstanpy", "cmdstanpy.model", "cmdstanpy.utils", "cmdstanpy.stanfit"):
#         lg = logging.getLogger(name)
#         lg.setLevel(logging.CRITICAL)
#         lg.propagate = False
#     os.environ["CMDSTANPY_LOG_LEVEL"] = "CRITICAL"
#     os.environ["CMDSTANPY_SILENT"] = "true"
#
#
# # =============================================================================
# # Small numerics helpers
# # =============================================================================
# def interp1_flat(x: np.ndarray, y: np.ndarray, xout: np.ndarray) -> np.ndarray:
#     """Linear interpolation with flat extrapolation (like R approx(rule=2))."""
#     x = np.asarray(x, float)
#     y = np.asarray(y, float)
#     xout = np.asarray(xout, float)
#
#     order = np.argsort(x)
#     x = x[order]
#     y = y[order]
#
#     yout = np.interp(xout, x, y)
#     yout = np.where(xout < x[0], y[0], yout)
#     yout = np.where(xout > x[-1], y[-1], yout)
#     return yout
#
#
# # =============================================================================
# # Optics LUTs (a_w, a0, a1) from CSV + bb_w formula
# # =============================================================================
# def load_optics_luts_from_csv(
#     wavelength: Sequence[float],
#     a_w_csv: str,
#     a0_a1_phyto_csv: str,
# ) -> Dict[str, np.ndarray]:
#     wl = np.asarray(wavelength, float)
#     if wl.ndim != 1 or wl.size < 2:
#         raise ValueError("wavelength must be 1D length>=2")
#
#     a_w_df = pd.read_csv(a_w_csv)
#     a01_df = pd.read_csv(a0_a1_phyto_csv)
#
#     for c in ("wavelength", "a_w"):
#         if c not in a_w_df.columns:
#             raise ValueError(f"{a_w_csv} missing column '{c}'")
#     for c in ("wavelength", "a0", "a1"):
#         if c not in a01_df.columns:
#             raise ValueError(f"{a0_a1_phyto_csv} missing column '{c}'")
#
#     a_w = interp1_flat(a_w_df["wavelength"].to_numpy(float), a_w_df["a_w"].to_numpy(float), wl)
#     a0 = interp1_flat(a01_df["wavelength"].to_numpy(float), a01_df["a0"].to_numpy(float), wl)
#     a1 = interp1_flat(a01_df["wavelength"].to_numpy(float), a01_df["a1"].to_numpy(float), wl)
#
#     # same bb_w parameterization you used
#     bb_w = 0.00111 * (wl / 500.0) ** (-4.32)
#
#     return {"a_w": a_w, "a0": a0, "a1": a1, "bb_w": bb_w}
#
#
# # =============================================================================
# # Compile Stan once (parent) and load exe in workers
# # =============================================================================
# def compile_model(stan_model_path: str, user_header: str | None = None) -> str:
#     from cmdstanpy import CmdStanModel
#     m = CmdStanModel(stan_file=stan_model_path, user_header=user_header)
#     return m.exe_file
#
#
# # =============================================================================
# # Worker init: build _BASE via the DATA BUILDER module
# # =============================================================================
# def _init_worker(
#     exe_file: str,
#     wavelength: np.ndarray,
#     *,
#     water_type: int,
#     shallow: int,
#     # LUT paths
#     a_w_csv: str,
#     a0_a1_phyto_csv: str,
#     # replicate spectra path
#     rb_obs_path: str,
#     # PCA/prior config
#     q: int,
#     tau_floor: float,
#     sigma2_floor: float,
#     # prior hyperparameters (must match Stan data names)
#     priors: dict,
# ) -> None:
#     global _STAN_MODEL, _BASE
#     silence_everything()
#
#     from cmdstanpy import CmdStanModel
#     # import your new builder module (PUT YOUR REAL MODULE PATH HERE)
#     from reverie.correction.aquatic.stan_data_gmm import make_stan_data_with_lowrank_pca
#
#     _STAN_MODEL = CmdStanModel(exe_file=exe_file)
#
#     wl = np.asarray(wavelength, float)
#
#     optics = load_optics_luts_from_csv(wl, a_w_csv=a_w_csv, a0_a1_phyto_csv=a0_a1_phyto_csv)
#
#     # build base with dummy obs; overwritten per segment
#     dummy_rrs = np.zeros_like(wl, dtype=float)
#     dummy_sig = np.full_like(wl, 1e-6, dtype=float)
#
#     base = make_stan_data_with_lowrank_pca(
#         wavelength=wl,
#         rrs_obs=dummy_rrs,
#         sigma_rrs=dummy_sig,
#         a_w=optics["a_w"],
#         a0=optics["a0"],
#         a1=optics["a1"],
#         bb_w=optics["bb_w"],
#         rb_obs_path=rb_obs_path,
#         q=q,
#         tau_floor=tau_floor,
#         sigma2_floor=sigma2_floor,
#         water_type=water_type,
#         theta_sun=30.0,      # overwritten per segment
#         theta_view=0.0,      # overwritten per segment
#         shallow=shallow,
#         # prior hypers
#         a_nap_star_mu=float(priors["a_nap_star_mu"]),
#         a_nap_star_sd=float(priors["a_nap_star_sd"]),
#         bb_p_star_mu=float(priors["bb_p_star_mu"]),
#         bb_p_star_sd=float(priors["bb_p_star_sd"]),
#         a_g_s_mu=float(priors["a_g_s_mu"]),
#         a_g_s_sd=float(priors["a_g_s_sd"]),
#         a_nap_s_mu=float(priors["a_nap_s_mu"]),
#         a_nap_s_sd=float(priors["a_nap_s_sd"]),
#         bb_p_gamma_mu=float(priors["bb_p_gamma_mu"]),
#         bb_p_gamma_sd=float(priors["bb_p_gamma_sd"]),
#         h_w_mu=float(priors["h_w_mu"]),
#         h_w_sd=float(priors["h_w_sd"]),
#     )
#
#     # keep base (includes rb_L_k etc). Per-segment we overwrite: rrs_obs, sigma_rrs, theta_sun/view, h_w_mu/sd
#     _BASE = base
#
#
# # =============================================================================
# # Segment inversion: optimize or sample
# # =============================================================================
# def invert_one_segment(job: dict, stan_cfg: dict, out_vars: tuple[str, ...]) -> dict:
#     global _STAN_MODEL, _BASE
#     assert _STAN_MODEL is not None
#     assert _BASE is not None
#
#     seg_id = int(job["seg_id"])
#     wl = job["wl_nm"].astype(float)
#
#     rho_est = job["rho_med"].astype(float)
#     rho_unc = job["rho_mad"].astype(float)
#
#     # rho -> rrs(0+) -> rrs(0-)
#     rrs_0p = rho_est / np.pi
#     rrs_0m = rrs_0p / (0.52 + 1.7 * rrs_0p)
#     if not np.all(np.isfinite(rrs_0m)):
#         return {"seg_id": seg_id, "ok": False, "err": "non-finite rrs_0m"}
#
#     rrs_0p_unc = rho_unc / np.pi
#     rrs_0m_unc = rrs_0p_unc / (0.52 + 1.7 * rrs_0p_unc)
#     rrs_0m_unc = np.maximum(rrs_0m_unc, 1e-8)
#
#     stan_data = dict(_BASE)
#     stan_data["rrs_obs"] = rrs_0m.tolist()
#     stan_data["sigma_rrs"] = rrs_0m_unc.tolist()
#     stan_data["theta_sun"] = float(job["theta_sun"])
#     stan_data["theta_view"] = float(job["theta_view"])
#     stan_data["h_w_mu"] = float(job["h_w_mu"])
#     stan_data["h_w_sd"] = float(job["h_w_sd"])
#
#     # quick sanity check: new model must have rb_L_k
#     if "rb_L_k" not in stan_data:
#         return {"seg_id": seg_id, "ok": False, "err": "stan_data missing rb_L_k (wrong builder/module?)"}
#
#     pid_dir = tempfile.mkdtemp(prefix="cmdstan_", dir="/tmp")
#
#     try:
#         if stan_cfg["method"] == "optimize":
#             fit = _STAN_MODEL.optimize(
#                 data=stan_data,
#                 seed=int(stan_cfg["seed"]),
#                 output_dir=pid_dir,
#                 show_console=False,
#             )
#             est = fit.optimized_params_dict
#         else:
#             fit = _STAN_MODEL.sample(
#                 data=stan_data,
#                 seed=int(stan_cfg["seed"]),
#                 chains=int(stan_cfg["chains"]),
#                 parallel_chains=1,
#                 iter_warmup=int(stan_cfg["iter_warmup"]),
#                 iter_sampling=int(stan_cfg["iter_sampling"]),
#                 show_progress=False,
#                 show_console=False,
#                 output_dir=pid_dir,
#             )
#             draws = fit.draws_pd()
#             est = {v: float(draws[v].mean()) for v in draws.columns}
#
#         scalars = {v: float(est[v]) for v in out_vars if v in est}
#
#         # Generated quantities (works for optimize if you declared them; otherwise may be absent)
#         nwl = len(wl)
#         r_b_hat = np.full(nwl, np.nan, dtype=float)
#         rrs_hat = np.full(nwl, np.nan, dtype=float)
#
#         for i in range(nwl):
#             kb = f"r_b_hat[{i + 1}]"
#             kr = f"rrs_hat[{i + 1}]"
#             if kb in est:
#                 r_b_hat[i] = float(est[kb])
#             if kr in est:
#                 rrs_hat[i] = float(est[kr])
#
#         # rrs(0-) -> rrs(0+) -> rho
#         rho_hat = np.full(nwl, np.nan, dtype=float)
#         ok = np.isfinite(rrs_hat) & (1 - 1.7 * rrs_hat > 1e-12)
#         rrs_0p_hat = np.full(nwl, np.nan, dtype=float)
#         rrs_0p_hat[ok] = (0.52 * rrs_hat[ok]) / (1 - 1.7 * rrs_hat[ok])
#         rho_hat[ok] = rrs_0p_hat[ok] * np.pi
#
#         r_b_hat = np.clip(np.nan_to_num(r_b_hat, nan=0.0), 0.0, 1.0)
#
#         return {
#             "seg_id": seg_id,
#             "ok": True,
#             "scalars": scalars,
#             "wl_nm": wl.astype(float),
#             "r_b_med": r_b_hat.astype(float),
#             "rho_hat": rho_hat.astype(float),
#         }
#
#     except Exception as e:
#         return {"seg_id": seg_id, "ok": False, "err": repr(e)}
#
#
# # =============================================================================
# # SQLite helpers (write non-spatial tables into GPKG)
# # =============================================================================
# def sql_safe(name: str) -> str:
#     return (
#         name.replace("[", "_")
#         .replace("]", "")
#         .replace(".", "_")
#         .replace("-", "_")
#         .replace(" ", "_")
#         .replace("(", "_")
#         .replace(")", "_")
#     )
#
#
# def write_table_sqlite(gpkg_path: str, table: str, df: pd.DataFrame, overwrite: bool = True) -> None:
#     conn = sqlite3.connect(gpkg_path)
#     cur = conn.cursor()
#     if overwrite:
#         cur.execute(f'DROP TABLE IF EXISTS "{table}"')
#
#     cols_def = []
#     for c in df.columns:
#         if np.issubdtype(df[c].dtype, np.integer):
#             t = "INTEGER"
#         elif np.issubdtype(df[c].dtype, np.floating):
#             t = "REAL"
#         else:
#             t = "TEXT"
#         cols_def.append(f'"{c}" {t}')
#
#     cur.execute(f'CREATE TABLE "{table}" ({", ".join(cols_def)})')
#
#     placeholders = ",".join(["?"] * len(df.columns))
#     cols = ", ".join(f'"{c}"' for c in df.columns)
#     ins = f'INSERT INTO "{table}" ({cols}) VALUES ({placeholders})'
#     cur.executemany(ins, df.itertuples(index=False, name=None))
#
#     conn.commit()
#     conn.close()
#
#
# # =============================================================================
# # Main driver: read segments.gpkg -> invert -> write segments_inverted.gpkg
# # =============================================================================
# def run_gpkg_inversion_to_new_file(
#     in_gpkg: str,
#     out_gpkg: str,
#     segments_layer: str = "segments",
#     spectra_table: str = "spectra",
#     aux_table: str = "aux",
#     seg_id_field: str = "seg_id",
#     wl_field: str = "wl_nm",
#     rho_med_field: str = "rho_med",
#     rho_mad_field: str = "rho_mad",
#     sun_field: str = "sun_zenith_med",
#     view_field: str = "view_zenith_med",
#     # IMPORTANT: these must exist in aux table, unless you override them
#     hw_mu_field: str = "h_w_mu",
#     hw_sd_field: str = "h_w_sd",
#     stan_model_path: str = "",
#     out_vars: tuple[str, ...] = ("chl", "a_g_440", "spm", "h_w", "sigma_model", "lp__"),
#     method: str = "optimize",
#     n_workers: int | None = None,
#     h_w_mu_override: float | None = None,
#     h_w_sd_override: float | None = None,
# ) -> int:
#     gdf = gpd.read_file(in_gpkg, layer=segments_layer)
#
#     conn = sqlite3.connect(in_gpkg)
#     spec = pd.read_sql_query(
#         f"""
#         SELECT
#             {seg_id_field}     AS seg_id,
#             {wl_field}         AS wl_nm,
#             {rho_med_field}    AS rho_med,
#             {rho_mad_field}    AS rho_mad
#         FROM "{spectra_table}"
#         """,
#         conn,
#     )
#     aux = pd.read_sql_query(
#         f"""
#         SELECT
#             {seg_id_field}   AS seg_id,
#             {sun_field}      AS theta_sun,
#             {view_field}     AS theta_view,
#             {hw_mu_field}    AS h_w_mu,
#             {hw_sd_field}    AS h_w_sd
#         FROM "{aux_table}"
#         """,
#         conn,
#     )
#     conn.close()
#
#     merged = spec.merge(aux, on="seg_id", how="inner")
#
#     jobs: list[dict] = []
#     for seg_id, gg in merged.groupby("seg_id"):
#         gg = gg.sort_values("wl_nm")
#
#         if not np.isfinite(gg["theta_sun"].iloc[0]) or not np.isfinite(gg["theta_view"].iloc[0]):
#             continue
#
#         if h_w_mu_override is None:
#             h_w_mu = float(gg["h_w_mu"].iloc[0])
#             if not np.isfinite(h_w_mu):
#                 continue
#         else:
#             h_w_mu = float(h_w_mu_override)
#
#         if h_w_sd_override is None:
#             h_w_sd = float(gg["h_w_sd"].iloc[0])
#             if not np.isfinite(h_w_sd):
#                 continue
#         else:
#             h_w_sd = float(h_w_sd_override)
#
#         jobs.append(
#             {
#                 "seg_id": int(seg_id),
#                 "wl_nm": gg["wl_nm"].to_numpy(float),
#                 "rho_med": gg["rho_med"].to_numpy(float),
#                 "rho_mad": gg["rho_mad"].to_numpy(float),
#                 "theta_sun": float(gg["theta_sun"].iloc[0]),
#                 "theta_view": float(gg["theta_view"].iloc[0]),
#                 "h_w_mu": h_w_mu,
#                 "h_w_sd": h_w_sd,
#             }
#         )
#
#     if not jobs:
#         raise RuntimeError("No valid segments to invert (check aux join / NaNs).")
#
#     if n_workers is None:
#         n_workers = max(1, (os.cpu_count() or 2) - 1)
#
#     wavelength = np.asarray(jobs[0]["wl_nm"], dtype=float)
#
#     # compile once
#     user_header = "/home/raphael/R/SABER/inst/stan/fct_rtm_stan.hpp"
#     exe_file = compile_model(stan_model_path, user_header=user_header)
#
#     # LUTs + replicate spectra
#     a_w_csv = "/home/raphael/R/SABER/inst/extdata/a_w.csv"
#     a0_a1_phyto_csv = "/home/raphael/R/SABER/inst/extdata/a0_a1_phyto.csv"
#     rb_obs_path = "/home/raphael/R/SABER/inst/extdata/r_b_gamache_obs.csv"
#
#     priors = dict(
#         a_nap_star_mu=0.0051, a_nap_star_sd=0.0012,
#         bb_p_star_mu=0.0047,  bb_p_star_sd=0.0012,
#         a_g_s_mu=0.017,       a_g_s_sd=0.0012,
#         a_nap_s_mu=0.006,     a_nap_s_sd=0.0012,
#         bb_p_gamma_mu=0.65,   bb_p_gamma_sd=0.12,
#         h_w_mu=3.0,           h_w_sd=2.0,   # default; overwritten per segment or override
#     )
#
#     initkwargs = dict(
#         water_type=2,
#         shallow=1,
#         a_w_csv=a_w_csv,
#         a0_a1_phyto_csv=a0_a1_phyto_csv,
#         rb_obs_path=rb_obs_path,
#         q=8,
#         tau_floor=1e-6,
#         sigma2_floor=1e-8,
#         priors=priors,
#     )
#
#     ctx = mp.get_context("spawn")
#     stan_cfg = dict(method=method, seed=1234, chains=2, iter_warmup=300, iter_sampling=300)
#
#     init_fn = functools.partial(_init_worker, **initkwargs)
#
#     results: list[dict] = []
#     failed: list[dict] = []
#
#     with ProcessPoolExecutor(
#         max_workers=n_workers,
#         mp_context=ctx,
#         initializer=init_fn,
#         initargs=(exe_file, wavelength),
#     ) as ex:
#         futs = [ex.submit(invert_one_segment, job, stan_cfg, out_vars) for job in jobs]
#         for fut in as_completed(futs):
#             r = fut.result()
#             if r.get("ok"):
#                 results.append(r)
#             else:
#                 failed.append(r)
#
#     if not results:
#         err = failed[0].get("err") if failed else "unknown"
#         raise RuntimeError(f"All inversions failed. Example error: {err}")
#
#     # outputs
#     scal_df = pd.DataFrame(
#         [{"seg_id": r["seg_id"], **{sql_safe(k): r["scalars"].get(k, np.nan) for k in out_vars}} for r in results]
#     )
#
#     gdf[seg_id_field] = gdf[seg_id_field].astype(int)
#     scal_df["seg_id"] = scal_df["seg_id"].astype(int)
#
#     gdf_out = gdf.merge(scal_df, left_on=seg_id_field, right_on="seg_id", how="left", suffixes=("", "_inv"))
#     if "seg_id_inv" in gdf_out.columns:
#         gdf_out = gdf_out.drop(columns=["seg_id_inv"])
#
#     rb_rows = []
#     for r in results:
#         seg_id = int(r["seg_id"])
#         wl_nm = r["wl_nm"]
#         rb_med = r["r_b_med"]
#         rho_hat = r["rho_hat"]
#
#         for wl_i, rb_i, rho_i in zip(wl_nm, rb_med, rho_hat):
#             rb_rows.append(
#                 {"seg_id": seg_id, "wl_nm": float(wl_i), "r_b_hat": float(rb_i), "rho_hat": float(rho_i)}
#             )
#     rb_df = pd.DataFrame(rb_rows)
#
#     out_dir = os.path.dirname(out_gpkg)
#     if out_dir:
#         os.makedirs(out_dir, exist_ok=True)
#     if os.path.exists(out_gpkg):
#         os.remove(out_gpkg)
#
#     gdf_out.to_file(out_gpkg, layer=segments_layer, driver="GPKG")
#     write_table_sqlite(out_gpkg, "spectra", rb_df, overwrite=True)
#
#     return len(results)
#
#
# # =============================================================================
# # Run
# # =============================================================================
# if __name__ == "__main__":
#     n_ok = run_gpkg_inversion_to_new_file(
#         in_gpkg="/D/Data/WISE/ACI-12A/el_sma/segmentation_outputs/segments.gpkg",
#         out_gpkg="/D/Data/WISE/ACI-12A/el_sma/segmentation_outputs/segments_inverted.gpkg",
#         segments_layer="segments",
#         spectra_table="spectra",
#         aux_table="aux",
#         stan_model_path="/home/raphael/R/SABER/inst/stan/model_gmm.stan",
#         out_vars=("chl", "a_g_440", "spm", "h_w", "sigma_model", "lp__"),
#         method="optimize",
#         n_workers=None,
#         h_w_mu_override=5,
#         h_w_sd_override=5,
#     )
#     print("segments inverted:", n_ok)

# --- PATCH: adapt your inversion script to the ETA model ---
# Assumptions:
# 1) Your Stan data block uses: eta_U, eta_mu_k, eta_L_k, eta_sigma2_k, eta_pi
# 2) Parameters include: vector[n_wl] eta_b; and transformed r_b = inv_logit(eta_b)
# 3) Generated quantities still include r_b_hat and rrs_hat (as in your Stan code)

from __future__ import annotations

import os
import logging
import sqlite3
import tempfile
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Optional, Sequence, Tuple
import functools

import numpy as np
import pandas as pd
import geopandas as gpd


# =============================================================================
# Logging / silence
# =============================================================================
_STAN_MODEL = None
_BASE: dict[str, Any] | None = None


def silence_everything() -> None:
    logging.disable(logging.ERROR)
    for name in ("cmdstanpy", "cmdstanpy.model", "cmdstanpy.utils", "cmdstanpy.stanfit"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.CRITICAL)
        lg.propagate = False
    os.environ["CMDSTANPY_LOG_LEVEL"] = "CRITICAL"
    os.environ["CMDSTANPY_SILENT"] = "true"


# =============================================================================
# Small numerics helpers
# =============================================================================
def interp1_flat(x: np.ndarray, y: np.ndarray, xout: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xout = np.asarray(xout, float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    yout = np.interp(xout, x, y)
    yout = np.where(xout < x[0], y[0], yout)
    yout = np.where(xout > x[-1], y[-1], yout)
    return yout


def load_optics_luts_from_csv(
    wavelength: Sequence[float],
    a_w_csv: str,
    a0_a1_phyto_csv: str,
) -> Dict[str, np.ndarray]:
    wl = np.asarray(wavelength, float)
    if wl.ndim != 1 or wl.size < 2:
        raise ValueError("wavelength must be 1D length>=2")

    a_w_df = pd.read_csv(a_w_csv)
    a01_df = pd.read_csv(a0_a1_phyto_csv)

    for c in ("wavelength", "a_w"):
        if c not in a_w_df.columns:
            raise ValueError(f"{a_w_csv} missing column '{c}'")
    for c in ("wavelength", "a0", "a1"):
        if c not in a01_df.columns:
            raise ValueError(f"{a0_a1_phyto_csv} missing column '{c}'")

    a_w = interp1_flat(a_w_df["wavelength"].to_numpy(float), a_w_df["a_w"].to_numpy(float), wl)
    a0 = interp1_flat(a01_df["wavelength"].to_numpy(float), a01_df["a0"].to_numpy(float), wl)
    a1 = interp1_flat(a01_df["wavelength"].to_numpy(float), a01_df["a1"].to_numpy(float), wl)

    bb_w = 0.00111 * (wl / 500.0) ** (-4.32)
    return {"a_w": a_w, "a0": a0, "a1": a1, "bb_w": bb_w}


def compile_model(stan_model_path: str, user_header: str | None = None) -> str:
    from cmdstanpy import CmdStanModel
    m = CmdStanModel(stan_file=stan_model_path, user_header=user_header)
    return m.exe_file


# =============================================================================
# Worker init: build _BASE for ETA model
# =============================================================================
def _init_worker(
    exe_file: str,
    wavelength: np.ndarray,
    *,
    water_type: int,
    shallow: int,
    a_w_csv: str,
    a0_a1_phyto_csv: str,
    rb_obs_path: str,
    # ETA prior config
    q: int,
    eps: float,
    cov_jitter: float,
    sigma2_floor: float,
    # prior hyperparameters (must match Stan data names)
    priors: dict,
) -> None:
    global _STAN_MODEL, _BASE
    silence_everything()

    from cmdstanpy import CmdStanModel
    # IMPORT the ETA builder you generated earlier (change module path as needed)
    from reverie.correction.aquatic.stan_data_gmm import make_stan_data_eta_model  # <-- adjust to your file/module name

    _STAN_MODEL = CmdStanModel(exe_file=exe_file)

    wl = np.asarray(wavelength, float)
    optics = load_optics_luts_from_csv(wl, a_w_csv=a_w_csv, a0_a1_phyto_csv=a0_a1_phyto_csv)

    dummy_rrs = np.zeros_like(wl, dtype=float)
    dummy_sig = np.full_like(wl, 1e-6, dtype=float)

    base = make_stan_data_eta_model(
        wavelength=wl,
        rrs_obs=dummy_rrs,
        sigma_rrs=dummy_sig,
        a_w=optics["a_w"],
        a0=optics["a0"],
        a1=optics["a1"],
        bb_w=optics["bb_w"],
        rb_obs_path=rb_obs_path,
        q=q,
        eps=eps,
        cov_jitter=cov_jitter,
        sigma2_floor=sigma2_floor,
        water_type=water_type,
        theta_sun=30.0,   # overwritten per segment
        theta_view=0.0,   # overwritten per segment
        shallow=shallow,
        # hypers
        a_nap_star_mu=float(priors["a_nap_star_mu"]),
        a_nap_star_sd=float(priors["a_nap_star_sd"]),
        bb_p_star_mu=float(priors["bb_p_star_mu"]),
        bb_p_star_sd=float(priors["bb_p_star_sd"]),
        a_g_s_mu=float(priors["a_g_s_mu"]),
        a_g_s_sd=float(priors["a_g_s_sd"]),
        a_nap_s_mu=float(priors["a_nap_s_mu"]),
        a_nap_s_sd=float(priors["a_nap_s_sd"]),
        bb_p_gamma_mu=float(priors["bb_p_gamma_mu"]),
        bb_p_gamma_sd=float(priors["bb_p_gamma_sd"]),
        h_w_mu=float(priors["h_w_mu"]),
        h_w_sd=float(priors["h_w_sd"]),
    )

    # Ensure eta-fields exist (defensive)
    required = ("eta_U", "eta_mu_k", "eta_L_k", "eta_sigma2_k", "eta_pi")
    miss = [k for k in required if k not in base]
    if miss:
        raise RuntimeError(f"ETA base missing keys: {miss}")

    _BASE = base


# =============================================================================
# Segment inversion
# =============================================================================
def invert_one_segment(job: dict, stan_cfg: dict, out_vars: tuple[str, ...]) -> dict:
    global _STAN_MODEL, _BASE
    assert _STAN_MODEL is not None
    assert _BASE is not None

    seg_id = int(job["seg_id"])
    wl = job["wl_nm"].astype(float)

    rho_est = job["rho_med"].astype(float)
    rho_unc = job["rho_mad"].astype(float)

    rrs_0p = rho_est / np.pi
    rrs_0m = rrs_0p / (0.52 + 1.7 * rrs_0p)
    if not np.all(np.isfinite(rrs_0m)):
        return {"seg_id": seg_id, "ok": False, "err": "non-finite rrs_0m"}

    rrs_0p_unc = rho_unc / np.pi
    rrs_0m_unc = rrs_0p_unc / (0.52 + 1.7 * rrs_0p_unc)
    rrs_0m_unc = np.maximum(rrs_0m_unc, 1e-8)

    stan_data = dict(_BASE)
    stan_data["rrs_obs"] = rrs_0m.tolist()
    stan_data["sigma_rrs"] = rrs_0m_unc.tolist()
    stan_data["theta_sun"] = float(job["theta_sun"])
    stan_data["theta_view"] = float(job["theta_view"])
    stan_data["h_w_mu"] = float(job["h_w_mu"])
    stan_data["h_w_sd"] = float(job["h_w_sd"])

    # sanity: eta fields must be present
    if "eta_U" not in stan_data:
        return {"seg_id": seg_id, "ok": False, "err": "stan_data missing eta_U (wrong base/builder?)"}

    pid_dir = tempfile.mkdtemp(prefix="cmdstan_", dir="/tmp")

    try:
        if stan_cfg["method"] == "optimize":
            fit = _STAN_MODEL.optimize(
                data=stan_data,
                seed=int(stan_cfg["seed"]),
                output_dir=pid_dir,
                show_console=False,
            )
            est = fit.optimized_params_dict
        else:
            fit = _STAN_MODEL.sample(
                data=stan_data,
                seed=int(stan_cfg["seed"]),
                chains=int(stan_cfg["chains"]),
                parallel_chains=1,
                iter_warmup=int(stan_cfg["iter_warmup"]),
                iter_sampling=int(stan_cfg["iter_sampling"]),
                show_progress=False,
                show_console=False,
                output_dir=pid_dir,
            )
            draws = fit.draws_pd()
            est = {v: float(draws[v].mean()) for v in draws.columns}

        scalars = {v: float(est[v]) for v in out_vars if v in est}

        nwl = len(wl)
        r_b_hat = np.full(nwl, np.nan, dtype=float)
        rrs_hat = np.full(nwl, np.nan, dtype=float)
        eta_b_hat = np.full(nwl, np.nan, dtype=float)

        for i in range(nwl):
            kb = f"r_b_hat[{i + 1}]"
            kr = f"rrs_hat[{i + 1}]"
            ke = f"eta_b[{i + 1}]"
            if kb in est:
                r_b_hat[i] = float(est[kb])
            if kr in est:
                rrs_hat[i] = float(est[kr])
            if ke in est:
                eta_b_hat[i] = float(est[ke])

        # rrs(0-) -> rrs(0+) -> rho
        rho_hat = np.full(nwl, np.nan, dtype=float)
        ok = np.isfinite(rrs_hat) & (1 - 1.7 * rrs_hat > 1e-12)
        rrs_0p_hat = np.full(nwl, np.nan, dtype=float)
        rrs_0p_hat[ok] = (0.52 * rrs_hat[ok]) / (1 - 1.7 * rrs_hat[ok])
        rho_hat[ok] = rrs_0p_hat[ok] * np.pi

        # physical reflectance already via inv_logit in Stan; keep it bounded
        r_b_hat = np.clip(np.nan_to_num(r_b_hat, nan=0.0), 0.0, 1.0)

        return {
            "seg_id": seg_id,
            "ok": True,
            "scalars": scalars,
            "wl_nm": wl.astype(float),
            "r_b_med": r_b_hat.astype(float),
            "rho_hat": rho_hat.astype(float),
            "eta_b_med": eta_b_hat.astype(float),  # optional debug output
        }

    except Exception as e:
        return {"seg_id": seg_id, "ok": False, "err": repr(e)}


# =============================================================================
# SQLite helpers
# =============================================================================
def sql_safe(name: str) -> str:
    return (
        name.replace("[", "_")
        .replace("]", "")
        .replace(".", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("(", "_")
        .replace(")", "_")
    )


def write_table_sqlite(gpkg_path: str, table: str, df: pd.DataFrame, overwrite: bool = True) -> None:
    conn = sqlite3.connect(gpkg_path)
    cur = conn.cursor()
    if overwrite:
        cur.execute(f'DROP TABLE IF EXISTS "{table}"')

    cols_def = []
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.integer):
            t = "INTEGER"
        elif np.issubdtype(df[c].dtype, np.floating):
            t = "REAL"
        else:
            t = "TEXT"
        cols_def.append(f'"{c}" {t}')

    cur.execute(f'CREATE TABLE "{table}" ({", ".join(cols_def)})')

    placeholders = ",".join(["?"] * len(df.columns))
    cols = ", ".join(f'"{c}"' for c in df.columns)
    ins = f'INSERT INTO "{table}" ({cols}) VALUES ({placeholders})'
    cur.executemany(ins, df.itertuples(index=False, name=None))

    conn.commit()
    conn.close()


def sqlite_table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    return {r[1] for r in rows}

def pick_existing_column(conn: sqlite3.Connection, table: str, candidates: Sequence[str]) -> str:
    cols = sqlite_table_columns(conn, table)
    for c in candidates:
        if c in cols:
            return c
    raise RuntimeError(
        f'None of these columns exist in "{table}": {list(candidates)}. Found: {sorted(cols)}'
    )

# =============================================================================
# Main driver (ONLY changes: initkwargs uses eps/cov_jitter; aux must provide h_w_mu/h_w_sd)
# =============================================================================
def run_gpkg_inversion_to_new_file(
    in_gpkg: str,
    out_gpkg: str,
    segments_layer: str = "segments",
    spectra_table: str = "spectra",
    seg_id_field: str = "seg_id",
    wl_field: str = "wl_nm",
    rho_med_field: str = "rho_med",
    rho_mad_field: str = "rho_mad",
    sun_field: str = "theta_sun",
    view_field: str = "theta_view",
    hw_mu_field: str = "h_w_mu",
    hw_sd_field: str = "h_w_sd",
    stan_model_path: str = "",
    out_vars: tuple[str, ...] = ("chl", "a_g_440", "spm", "h_w", "sigma_model", "lp__"),
    method: str = "optimize",
    n_workers: int | None = None,
    h_w_mu_override: float | None = None,
    h_w_sd_override: float | None = None,
) -> int:
    # segments layer (now contains the scalar fields)
    gdf = gpd.read_file(in_gpkg, layer=segments_layer)
    if seg_id_field not in gdf.columns:
        raise RuntimeError(f"'{seg_id_field}' not found in segments layer")

    for c in (sun_field, view_field, hw_mu_field, hw_sd_field):
        if c not in gdf.columns:
            raise RuntimeError(f"'{c}' not found in segments layer. Available: {list(gdf.columns)}")

    # read spectra table (SQLite) from the same gpkg
    conn = sqlite3.connect(in_gpkg)
    spec = pd.read_sql_query(
        f"""
        SELECT
            {seg_id_field}   AS seg_id,
            {wl_field}       AS wl_nm,
            {rho_med_field}  AS rho_med,
            {rho_mad_field}  AS rho_mad
        FROM "{spectra_table}"
        """,
        conn,
    )
    conn.close()

    # build a scalars dataframe from the segments layer (no aux table anymore)
    seg_scal = pd.DataFrame(
        {
            "seg_id": pd.to_numeric(gdf[seg_id_field], errors="coerce").astype("Int64"),
            "theta_sun": pd.to_numeric(gdf[sun_field], errors="coerce"),
            "theta_view": pd.to_numeric(gdf[view_field], errors="coerce"),
            "h_w_mu": pd.to_numeric(gdf[hw_mu_field], errors="coerce"),
            "h_w_sd": pd.to_numeric(gdf[hw_sd_field], errors="coerce"),
        }
    ).dropna(subset=["seg_id"])
    seg_scal["seg_id"] = seg_scal["seg_id"].astype(int)

    # merge spectra + scalars
    spec["seg_id"] = pd.to_numeric(spec["seg_id"], errors="coerce")
    spec = spec.dropna(subset=["seg_id"])
    spec["seg_id"] = spec["seg_id"].astype(int)

    merged = spec.merge(seg_scal, on="seg_id", how="inner")

    jobs: list[dict] = []
    for seg_id, gg in merged.groupby("seg_id"):
        gg = gg.sort_values("wl_nm")

        theta_sun = float(gg["theta_sun"].iloc[0])
        theta_view = float(gg["theta_view"].iloc[0])

        if not np.isfinite(theta_sun) or not np.isfinite(theta_view):
            continue

        if h_w_mu_override is None:
            h_w_mu = float(gg["h_w_mu"].iloc[0])
            if not np.isfinite(h_w_mu):
                continue
        else:
            h_w_mu = float(h_w_mu_override)

        if h_w_sd_override is None:
            h_w_sd = float(gg["h_w_sd"].iloc[0])
            if not np.isfinite(h_w_sd):
                continue
        else:
            h_w_sd = float(h_w_sd_override)

        jobs.append(
            {
                "seg_id": int(seg_id),
                "wl_nm": gg["wl_nm"].to_numpy(float),
                "rho_med": gg["rho_med"].to_numpy(float),
                "rho_mad": gg["rho_mad"].to_numpy(float),
                "theta_sun": theta_sun,
                "theta_view": theta_view,
                "h_w_mu": h_w_mu,
                "h_w_sd": h_w_sd,
            }
        )

    if not jobs:
        raise RuntimeError("No valid segments to invert (check NaNs in segments scalars or join).")

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    wavelength = np.asarray(jobs[0]["wl_nm"], dtype=float)

    user_header = "/home/raphael/R/SABER/inst/stan/fct_rtm_stan.hpp"
    exe_file = compile_model(stan_model_path, user_header=user_header)

    a_w_csv = "/home/raphael/R/SABER/inst/extdata/a_w.csv"
    a0_a1_phyto_csv = "/home/raphael/R/SABER/inst/extdata/a0_a1_phyto.csv"
    rb_obs_path = "/home/raphael/R/SABER/inst/extdata/r_b_gamache_obs.csv"

    priors = dict(
        a_nap_star_mu=0.0051, a_nap_star_sd=0.0012,
        bb_p_star_mu=0.0047,  bb_p_star_sd=0.0012,
        a_g_s_mu=0.017,       a_g_s_sd=0.0012,
        a_nap_s_mu=0.006,     a_nap_s_sd=0.0012,
        bb_p_gamma_mu=0.65,   bb_p_gamma_sd=0.12,
        h_w_mu=3.0,           h_w_sd=2.0,
    )

    initkwargs = dict(
        water_type=2,
        shallow=1,
        a_w_csv=a_w_csv,
        a0_a1_phyto_csv=a0_a1_phyto_csv,
        rb_obs_path=rb_obs_path,
        q=5,
        eps=1e-6,
        cov_jitter=1e-6,
        sigma2_floor=1e-6,
        priors=priors,
    )

    ctx = mp.get_context("spawn")
    stan_cfg = dict(method=method, seed=1234, chains=2, iter_warmup=300, iter_sampling=300)

    init_fn = functools.partial(_init_worker, **initkwargs)

    results: list[dict] = []
    failed: list[dict] = []

    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=init_fn,
        initargs=(exe_file, wavelength),
    ) as ex:
        futs = [ex.submit(invert_one_segment, job, stan_cfg, out_vars) for job in jobs]
        for fut in as_completed(futs):
            r = fut.result()
            if r.get("ok"):
                results.append(r)
            else:
                failed.append(r)

    if not results:
        err = failed[0].get("err") if failed else "unknown"
        raise RuntimeError(f"All inversions failed. Example error: {err}")

    scal_df = pd.DataFrame(
        [{"seg_id": r["seg_id"], **{sql_safe(k): r["scalars"].get(k, np.nan) for k in out_vars}} for r in results]
    )

    gdf[seg_id_field] = pd.to_numeric(gdf[seg_id_field], errors="coerce")
    gdf = gdf.dropna(subset=[seg_id_field])
    gdf[seg_id_field] = gdf[seg_id_field].astype(int)
    scal_df["seg_id"] = scal_df["seg_id"].astype(int)

    gdf_out = gdf.merge(scal_df, left_on=seg_id_field, right_on="seg_id", how="left", suffixes=("", "_inv"))
    if "seg_id_inv" in gdf_out.columns:
        gdf_out = gdf_out.drop(columns=["seg_id_inv"])

    rb_rows = []
    for r in results:
        seg_id = int(r["seg_id"])
        wl_nm = r["wl_nm"]
        rb_med = r["r_b_med"]
        rho_hat = r["rho_hat"]
        eta_med = r.get("eta_b_med", None)

        for j, (wl_i, rb_i, rho_i) in enumerate(zip(wl_nm, rb_med, rho_hat)):
            row = {"seg_id": seg_id, "wl_nm": float(wl_i), "r_b_hat": float(rb_i), "rho_hat": float(rho_i)}
            if eta_med is not None and j < len(eta_med) and np.isfinite(eta_med[j]):
                row["eta_b_hat"] = float(eta_med[j])
            rb_rows.append(row)

    rb_df = pd.DataFrame(rb_rows)

    out_dir = os.path.dirname(out_gpkg)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(out_gpkg):
        os.remove(out_gpkg)

    gdf_out.to_file(out_gpkg, layer=segments_layer, driver="GPKG")
    write_table_sqlite(out_gpkg, "spectra", rb_df, overwrite=True)

    return (len(results), failed)

# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    n_ok, failed = run_gpkg_inversion_to_new_file(
        in_gpkg="/D/Data/WISE/ACI-12A/el_sma/segmentation_outputs/segments.gpkg",
        out_gpkg="/D/Data/WISE/ACI-12A/el_sma/segmentation_outputs/segments_inverted.gpkg",
        segments_layer="segments",
        spectra_table="spectra",
        stan_model_path="/home/raphael/R/SABER/inst/stan/model_gmm.stan",
        out_vars=("chl", "a_g_440", "spm", "h_w", "sigma_model", "lp__"),
        # method="optimize",
        method="sample",
        n_workers=None,
        h_w_mu_override=5,
        h_w_sd_override=5,
        # h_w_mu_override=None,
        # h_w_sd_override=0.1,
    )
    print("segments inverted:", n_ok)
    print("segments failed:", len(failed))
    # print(failed[0].get("err"))

# from qgis.core import QgsProject, QgsVectorLayer
#
# seg_id = [% "seg_id" %]
#
# proj = QgsProject.instance()
#
# tbl = proj.mapLayersByName("segments_inverted — spectra")[0]  # <-- exact layer name
# tbl.removeSelection()
#
# # build expression depending on seg_id type
# if isinstance(seg_id, (int, float)):
#     expr = f"\"seg_id\" = {int(seg_id)}"
# else:
#     expr = f"\"seg_id\" = '{seg_id}'"
#
# tbl.selectByExpression(expr, QgsVectorLayer.SetSelection)
