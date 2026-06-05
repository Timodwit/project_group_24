"""
Magnetic Tweezers — Section 3: Data Analysis & Interpretation
VU Amsterdam Biophysics Practicum 2026
Authors: fill in your names

Dependencies:  numpy, scipy, pandas, matplotlib
Install:       pip install numpy scipy pandas matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq, curve_fit
from pathlib import Path

plt.rcParams.update({'font.size': 10, 'figure.dpi': 120})

# ------------------------------------------------------------------------------
# 0.  PHYSICAL CONSTANTS
# ------------------------------------------------------------------------------
kB       = 1.38e-23          # J/K
T        = 293.15            # K  (20 °C)
kBT_J    = kB * T            # ~4.1e-21 J
kBT_pNnm = kBT_J * 1e21      # ~4.1 pN·nm  (useful check: 1 kBT ~ 4.1 pN·nm)

eta      = 1e-3              # Pa·s   (water viscosity)
r_m      = 0.5e-6            # m      (bead radius 0.5 um)
gamma    = 6 * np.pi * eta * r_m   # Stokes drag  [N·s/m]

L_p      = 45e-9             # m      (persistence length DNA)
N_bp     = 20_600            # base pairs
L_c_m    = N_bp * 0.34e-9   # m      (contour length)

f_ac     = 10.0              # Hz     (camera acquisition frequency for Part 2)

# ------------------------------------------------------------------------------
# 1.  PHYSICS HELPERS
# ------------------------------------------------------------------------------

def wlc_force(L_ext_m: float) -> float:
    """WLC  Eq. 12:  F(L_ext) [pN].  Marko-Siggia formula."""
    x = np.clip(L_ext_m / L_c_m, 1e-9, 0.99999)
    F_N = (kBT_J / L_p) * (1.0 / (4.0*(1.0 - x)**2) - 0.25 + x)
    return F_N * 1e12  # pN


def wlc_L_ext(F_pN: float) -> float:
    """Invert WLC:  F [pN]  ->  L_ext [um]."""
    F_N = F_pN * 1e-12
    def residual(L):
        return wlc_force(L) * 1e-12 - F_N
    L_m = brentq(residual, 1e-12, 0.99999 * L_c_m)
    return L_m * 1e6   # um


def t_cx(F_pN: float, L_ext_um: float) -> float:
    """Characteristic time  t_{c,x} = γ L_ext / F  [s]   Eq. 7."""
    return gamma * (L_ext_um * 1e-6) / (F_pN * 1e-12)


def var_x_theory(F_pN: float, L_ext_um: float) -> float:
    """Theoretical  <x^2>  [nm²]  from Eq. 8  (axis _|_ B)."""
    return kBT_J * (L_ext_um * 1e-6) / (F_pN * 1e-12) * 1e18


def var_y_theory(F_pN: float, L_ext_um: float) -> float:
    """Theoretical  <y^2>  [nm²]  from Eq. 9  (axis // B, includes bead radius)."""
    return kBT_J * (L_ext_um * 1e-6 + r_m) / (F_pN * 1e-12) * 1e18


def force_eq8(var_nm2: float, L_ext_um: float) -> float:
    """Force from Eq. 8   (use on axis _|_ B):   F = kBT·L_ext / <x^2>   [pN]."""
    return kBT_J * (L_ext_um * 1e-6) / (var_nm2 * 1e-18) * 1e12


def force_eq9(var_nm2: float, L_ext_um: float) -> float:
    """Force from Eq. 9   (use on axis // B):   F = kBT·(L_ext+r) / <y^2>   [pN]."""
    return kBT_J * (L_ext_um * 1e-6 + r_m) / (var_nm2 * 1e-18) * 1e12


def shutter_overestimation(tcx_s: float, tau_sh_s: float) -> float:
    """
    F_meas / F_true = π / (2·arctan(π·t_{c,x}/tau_sh))
    Returns the overestimation factor (≥ 1).  Derived from Eq. 11.
    """
    u = np.pi * tcx_s / tau_sh_s
    return np.pi / (2.0 * np.arctan(u))


def F_z(z_mm):
    """Magnet force-distance calibration curve  [pN]  (Preparatory Q4)."""
    return 22.3811 * np.exp(-1.4578 * z_mm) + 52.2987 * np.exp(-0.6912 * z_mm)


def z_from_F(F_pN: float) -> float:
    """Invert F(z):  force [pN]  ->  magnet position z [mm]."""
    return brentq(lambda z: F_z(z) - F_pN, 0.01, 20.0)


# ------------------------------------------------------------------------------
# 2.  DATA I/O & UTILITIES
# ------------------------------------------------------------------------------

def load_data(path_or_df):
    """
    Read a tab-delimited MT data file, or pass through a DataFrame directly.

    Expected column order:
        frame | time_ms | x1 y1 z1 | x2 y2 z2 | … (positions in um)
        Last two bead triplets are reference beads N1, N2.

    Returns
    -------
    df      : DataFrame with named columns
    n_beads : total number of beads (tethered + 2 reference)
    """
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
        n_beads = (df.shape[1] - 2) // 3
        return df, n_beads
    raw = pd.read_csv(path_or_df, sep='\t', header=None, comment='#')
    n_beads = (raw.shape[1] - 2) // 3
    # Drop any trailing extra columns beyond the expected triplets
    n_cols = 2 + n_beads * 3
    raw = raw.iloc[:, :n_cols]
    cols = ['frame', 'time_ms']
    for i in range(1, n_beads + 1):
        cols += [f'x{i}', f'y{i}', f'z{i}']
    raw.columns = cols
    return raw, n_beads


def pop_variance(arr: np.ndarray) -> float:
    """Population variance  sigma^2 = (1/N) Σ(xᵢ − x̄)²   (Eq. 13)."""
    return float(np.sum((arr - arr.mean())**2) / len(arr))


def sigma_clip(arr: np.ndarray, n_sigma: float = 5.0):
    """
    Remove spikes beyond n_sigma·std.
    Returns (clipped_array, boolean_mask).
    """
    mu  = arr.mean()
    sig = arr.std()
    mask = np.abs(arr - mu) < n_sigma * sig
    return arr[mask], mask


# ------------------------------------------------------------------------------
# 3.  PART 1 — TRACKING RESOLUTION  (Measurement §1)
# ------------------------------------------------------------------------------

def part1_tracking_resolution(file_200gl: str, file_20gl: str):
    """
    Evaluate tracking noise from stuck reference beads.

    Steps
    -----
    1. Variance (x,y,z) of N1 and N2 separately — both illumination levels.
    2. Drift-corrected variance: N2 − N1  (single-bead noise = result / 2).
    3. Compare single-bead noise to theoretical <x^2> at 6 pN.
    4. Plot position traces.
    """
    L6    = wlc_L_ext(6.0)               # L_ext at 6 pN  [um]
    v6_th = var_x_theory(6.0, L6)        # theoretical <x^2> at 6 pN  [nm²]

    print(f"\n{'='*60}")
    print(f"  PART 1 — TRACKING RESOLUTION")
    print(f"  Reference:  <x^2>_theory at 6 pN = {v6_th:.1f} nm²  "
          f"(L_ext = {L6:.2f} um)")
    print(f"{'='*60}")

    results = {}

    for label, fpath in [('200 GL', file_200gl), ('20 GL', file_20gl)]:
        df, n_beads = load_data(fpath)
        n1 = n_beads - 1   # 1-based index of N1
        n2 = n_beads       # 1-based index of N2

        print(f"\n  -- {label} --")
        print(f"  {'Bead':<6} {'axis':<5} {'sigma^2 [nm²]':>12}  {'σ [nm]':>10}")

        per_bead = {}
        for name, idx in [('N1', n1), ('N2', n2)]:
            variances = {}
            for ax in ('x', 'y', 'z'):
                arr = df[f'{ax}{idx}'].values * 1e3  # um -> nm
                v   = pop_variance(arr)
                variances[ax] = v
                print(f"  {name:<6} {ax:<5} {v:>12.2f}  {np.sqrt(v):>10.2f}")
            per_bead[name] = variances

        # Drift-corrected  N2 − N1
        print(f"\n  Drift-corrected (N2 − N1):")
        print(f"  {'axis':<5} {'sigma^2_corr':>12}  {'sigma^2_single':>12}  "
              f"{'ratio to 6pN theory':>22}")
        drift_corr = {}
        for ax in ('x', 'y', 'z'):
            n1_arr = df[f'{ax}{n1}'].values * 1e3
            n2_arr = df[f'{ax}{n2}'].values * 1e3
            diff   = n2_arr - n1_arr
            v_corr   = pop_variance(diff)
            v_single = v_corr / 2.0   # both beads contribute equally
            ratio    = v_single / v6_th
            drift_corr[ax] = v_single
            print(f"  {ax:<5} {v_corr:>12.2f}  {v_single:>12.2f}  "
                  f"  {ratio:>19.4f}×")

        results[label] = {'per_bead': per_bead, 'drift_corr': drift_corr,
                          'v6_theory': v6_th}

        # -- Plots ----------------------------------------------------------
        t      = df['time_ms'].values / 1e3   # s
        n1x_nm = df[f'x{n1}'].values * 1e3
        n2x_nm = df[f'x{n2}'].values * 1e3
        n1z_nm = df[f'z{n1}'].values * 1e3
        n2z_nm = df[f'z{n2}'].values * 1e3
        diff_x = n2x_nm - n1x_nm

        fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

        axes[0].plot(t, n1x_nm, lw=0.6, label='N1', alpha=0.8)
        axes[0].plot(t, n2x_nm, lw=0.6, label='N2', alpha=0.8)
        axes[0].set_ylabel('x  [nm]')
        axes[0].legend(fontsize=8)
        axes[0].set_title(f'Raw x-positions — {label}')

        axes[1].plot(t, diff_x, lw=0.6, color='seagreen', label='N2 − N1')
        axes[1].axhline(0, color='k', lw=0.5, ls='--')
        axes[1].set_ylabel('Δx  [nm]')
        axes[1].legend(fontsize=8)
        axes[1].set_title('Drift-corrected x')

        axes[2].plot(t, n1z_nm, lw=0.6, label='N1', alpha=0.8)
        axes[2].plot(t, n2z_nm, lw=0.6, label='N2', alpha=0.8)
        axes[2].set_ylabel('z  [nm]')
        axes[2].set_xlabel('Time  [s]')
        axes[2].legend(fontsize=8)
        axes[2].set_title('z-positions (drift visible here)')

        plt.suptitle(f'Reference bead traces — {label}', fontsize=11, y=1.01)
        plt.tight_layout()
        out = f"tracking_{label.replace(' ', '_')}.pdf"
        plt.savefig(out, bbox_inches='tight')
        plt.show()
        print(f"  -> saved {out}")

    return results


# ------------------------------------------------------------------------------
# 4.  PART 2 — FORCE CALIBRATION CORE
# ------------------------------------------------------------------------------

SET_FORCES = [0.1, 0.2, 0.3, 0.4, 0.8, 1.0, 3.0, 5.0, 6.0]   # pN


def _analyze_bead(df: pd.DataFrame,
                  bead_idx: int,
                  ref_idx: int,
                  n_zero: int) -> dict:
    """
    Extract L_ext, variances, and all force estimates for one tethered bead.

    Parameters
    ----------
    df        : DataFrame for one force-step file
    bead_idx  : 1-based index of the tethered bead
    ref_idx   : 1-based index of the reference bead (drift correction)
    n_zero    : number of frames at zero force at the beginning

    Returns dict with L_ext, variances, four force estimates
    """
    # Drift correction
    for ax in ('x', 'y', 'z'):
        df[f'd{ax}'] = (df[f'{ax}{bead_idx}'] - df[f'{ax}{ref_idx}']) * 1e3  # nm

    # L_ext from z: |median(z_force) − median(z_zero)|
    z_all   = df['dz'].values
    z_zero  = z_all[:n_zero]
    z_force = z_all[n_zero:]
    L_ext_nm = abs(np.median(z_force) - np.median(z_zero))
    L_ext_um = L_ext_nm / 1e3

    # Lateral variances in force region (sigma-clip spikes first)
    x_raw = df['dx'].values[n_zero:]
    y_raw = df['dy'].values[n_zero:]
    x_c, _ = sigma_clip(x_raw)
    y_c, _ = sigma_clip(y_raw)
    var_x = pop_variance(x_c)   # nm²
    var_y = pop_variance(y_c)   # nm²

    # Force estimates — apply Eq. 8 to both axes
    Fx8 = force_eq8(var_x, L_ext_um)
    Fy8 = force_eq8(var_y, L_ext_um)
    # Eq. 9 accounts for bead radius (correct for the axis // B)
    Fx9 = force_eq9(var_x, L_ext_um)
    Fy9 = force_eq9(var_y, L_ext_um)

    return dict(L_ext_um=L_ext_um,
                var_x=var_x, var_y=var_y,
                Fx_eq8=Fx8, Fy_eq8=Fy8,
                Fx_eq9=Fx9, Fy_eq9=Fy9)


def process_dataset(files_dict: dict,
                    tethered: list,
                    ref_idx: int,
                    n_zero: int,
                    tau_sh_label: str,
                    exclude_beads: list = None) -> pd.DataFrame:
    """
    Process a full tau_sh dataset (all force steps, all beads).

    Identifies the axis parallel to B:
      Using only Eq. 8, the axis // B gives a *lower* apparent force
      (because its effective arm is L_ext + r > L_ext, inflating the variance
       and deflating the Eq. 8 force estimate).
      Once identified, the correct force on that axis uses Eq. 9.

    Returns a tidy DataFrame with one row per (bead, force_set).
    """
    records = []

    for F_set in sorted(files_dict):
        val = files_dict[F_set]
        df, _ = load_data(val)

        for b in tethered:
            res = _analyze_bead(df.copy(), b, ref_idx, n_zero)

            # -- Axis identification --------------------------------------
            # Lower Eq. 8 force  ←->  larger variance  ←->  axis // B
            if res['Fx_eq8'] < res['Fy_eq8']:
                F_par  = res['Fx_eq9']   # x is // B  -> correct with Eq. 9
                axis_B = 'x'
            else:
                F_par  = res['Fy_eq9']   # y is // B
                axis_B = 'y'

            records.append({
                'tau_sh':    tau_sh_label,
                'F_set_pN':  F_set,
                'bead':      b,
                'L_ext_um':  res['L_ext_um'],
                'var_x_nm2': res['var_x'],
                'var_y_nm2': res['var_y'],
                'Fx_eq8':    res['Fx_eq8'],
                'Fy_eq8':    res['Fy_eq8'],
                'Fx_eq9':    res['Fx_eq9'],
                'Fy_eq9':    res['Fy_eq9'],
                'F_par_pN':  F_par,
                'axis_B':    axis_B,
            })

    df_out = pd.DataFrame(records)

    # Print summary table (excluding beads flagged for omission)
    _excl = set(exclude_beads) if exclude_beads else set()
    df_calc = df_out[~df_out['bead'].isin(_excl)]

    print(f"\n{'-'*62}")
    print(f"  tau_sh = {tau_sh_label}"
          + (f"  (excluded from stats: beads {sorted(_excl)})" if _excl else ""))
    print(f"{'-'*62}")
    print(f"  {'F_set':>7}  {'F_mean_par':>8}  {'sigma_F':>7}  {'rel%':>7}  n")
    for F_set, grp in df_calc.groupby('F_set_pN'):
        μ   = grp['F_par_pN'].mean()
        σ   = grp['F_par_pN'].std()
        rel = σ / μ * 100 if μ > 0 else 0
        print(f"  {F_set:>7.2f}  {μ:>8.3f}  {σ:>7.3f}  {rel:>6.1f}%  {len(grp)}")

    return df_out


# ------------------------------------------------------------------------------
# 5.  PLOTS (Parts 2.1 – 2.3)
# ------------------------------------------------------------------------------

def plot_F_Lext(df: pd.DataFrame, tau_sh_label: str):
    """F_par vs L_ext with WLC overlay  (per bead)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    cmap = plt.get_cmap('tab20')
    for i, (b, grp) in enumerate(df.groupby('bead')):
        ax.scatter(grp['L_ext_um'], grp['F_par_pN'],
                   color=cmap(i), s=35, label=f'Bead {b}', zorder=3)

    F_wlc = np.logspace(-1.5, 0.9, 300)
    L_wlc = np.array([wlc_L_ext(f) for f in F_wlc])
    ax.plot(L_wlc, F_wlc, 'k--', lw=1.8, label='WLC theory', zorder=4)

    ax.set_xlabel(r'$L_\mathrm{ext}$  [$\mu$m]')
    ax.set_ylabel('Force  [pN]')
    ax.set_title(f'Force vs extension — $\\tau_{{sh}}$ = {tau_sh_label}')
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = f"F_Lext_{tau_sh_label.replace(' ','_')}.pdf"
    plt.savefig(out, bbox_inches='tight')
    plt.show()
    print(f"  -> saved {out}")


def plot_F_z_combined(dfs: dict):
    """F_par vs magnet position z — fit + theory curve."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Theory curve
    z_arr = np.linspace(0.1, 13, 400)
    ax.plot(z_arr, F_z(z_arr), 'k--', lw=1.5, label='F(z) theory', zorder=5)

    colors = {'1 ms': 'steelblue', '10 ms': 'darkorange', '39 ms': 'seagreen',
              '0.4 ms': 'steelblue'}

    def _double_exp(z, a, b, c, d):
        return a * np.exp(-b * z) + c * np.exp(-d * z)

    for tau_label, df in dfs.items():
        col = colors.get(tau_label, 'gray')

        # Collect (z_mag, mean F_par) per force step for the fit
        z_pts, F_pts = [], []
        first = True
        for F_set, grp in df.groupby('F_set_pN'):
            z_mag  = z_from_F(F_set)
            F_vals = grp['F_par_pN'].values
            lbl    = f'tau_sh = {tau_label}' if first else None
            ax.scatter([z_mag]*len(F_vals), F_vals,
                       color=col, s=22, alpha=0.75, label=lbl, zorder=3)
            z_pts.append(z_mag)
            F_pts.append(np.nanmean(F_vals[np.isfinite(F_vals)]))
            first = False

        # Fit double-exponential to the per-step means (drop NaN/inf)
        z_pts = np.array(z_pts)
        F_pts = np.array(F_pts)
        ok = np.isfinite(F_pts) & np.isfinite(z_pts)
        z_pts, F_pts = z_pts[ok], F_pts[ok]
        try:
            p0 = [22.4, 1.46, 52.3, 0.69]   # start from theory values
            popt, _ = curve_fit(_double_exp, z_pts, F_pts, p0=p0,
                                bounds=(0, np.inf), maxfev=10000)
            z_fit = np.linspace(z_pts.min() * 0.95, z_pts.max() * 1.05, 400)
            ax.plot(z_fit, _double_exp(z_fit, *popt), color='crimson', lw=3,
                    label=(f'fit {tau_label}: '
                           f'${popt[0]:.2f}\\,e^{{-{popt[1]:.3f}z}}'
                           f' + {popt[2]:.2f}\\,e^{{-{popt[3]:.3f}z}}$'), zorder=4)
            print(f"  Fit ({tau_label}): F(z) = {popt[0]:.3f}*exp(-{popt[1]:.4f}*z)"
                  f" + {popt[2]:.3f}*exp(-{popt[3]:.4f}*z)")
        except RuntimeError as e:
            print(f"  Fit failed for {tau_label}: {e}")

    ax.set_xlabel('Magnet position  $z$  [mm]')
    ax.set_ylabel('Force  [pN]')
    ax.set_title('Force vs magnet position — fit vs theory')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('F_z_combined.pdf', bbox_inches='tight')
    plt.show()
    print("  -> saved F_z_combined.pdf")


def plot_axis_identification(df: pd.DataFrame, tau_sh_label: str):
    """
    Show Fx_eq8 and Fy_eq8 vs F_set on a log-log plot.
    The axis // B will systematically underestimate with Eq. 8
    (sits below the F_set diagonal).
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap('tab20')

    for i, (b, grp) in enumerate(df.groupby('bead')):
        c = cmap(i)
        ax.scatter(grp['F_set_pN'], grp['Fx_eq8'],
                   color=c, marker='o', s=30, label=f'B{b} x (Eq.8)')
        ax.scatter(grp['F_set_pN'], grp['Fy_eq8'],
                   color=c, marker='s', s=30, alpha=0.6, label=f'B{b} y (Eq.8)')

    f_rng = [df['F_set_pN'].min()*0.8, df['F_set_pN'].max()*1.2]
    ax.plot(f_rng, f_rng, 'k-', lw=1, label='F_set (ideal)')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$F_\mathrm{set}$  [pN]')
    ax.set_ylabel(r'$F_\mathrm{measured}$ (Eq. 8)  [pN]')
    ax.set_title(f'Axis $\\parallel$ B identification — $\\tau_{{sh}}$ = {tau_sh_label}\n'
                 f'($\\parallel$ B axis gives lower Eq. 8 force; correct with Eq. 9)')
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    out = f"axis_id_{tau_sh_label.replace(' ','_')}.pdf"
    plt.savefig(out, bbox_inches='tight')
    plt.show()
    print(f"  -> saved {out}")


# ------------------------------------------------------------------------------
# 6.  SHUTTER-TIME NORMALIZATION  (Analysis step 4)
# ------------------------------------------------------------------------------

def shutter_normalization(df_1ms: pd.DataFrame,
                          df_10ms: pd.DataFrame,
                          df_39ms: pd.DataFrame):
    """
    Normalize F(10 ms) and F(39 ms) by F(1 ms) per bead per force.
    Plot against  u = t_{c,x} / tau_sh  and overlay theory.

    Theory (from Eq. 11):
        F_meas(tau_sh) / F_true  ~  π / (2·arctan(π·u))
    where  u = t_{c,x} / tau_sh.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'10 ms': 'darkorange', '39 ms': 'seagreen'}
    tau_map = {'10 ms': 10e-3, '39 ms': 39e-3}

    for tau_label, df_tau in [('10 ms', df_10ms), ('39 ms', df_39ms)]:
        tau = tau_map[tau_label]
        col = colors[tau_label]
        first = True

        for b in df_1ms['bead'].unique():
            ref_b = (df_1ms[df_1ms['bead'] == b]
                     .set_index('F_set_pN')['F_par_pN'])
            tau_b = (df_tau[df_tau['bead'] == b]
                     .set_index('F_set_pN')['F_par_pN'])
            Lext_b = (df_1ms[df_1ms['bead'] == b]
                      .set_index('F_set_pN')['L_ext_um'])

            for F_set in ref_b.index:
                if F_set not in tau_b.index:
                    continue
                F_ref  = ref_b[F_set]
                F_tau  = tau_b[F_set]
                L_ext  = Lext_b[F_set]
                tcx    = t_cx(F_set, L_ext)
                u      = tcx / tau
                ratio  = F_tau / F_ref

                lbl = f'tau_sh = {tau_label}' if first else None
                ax.scatter(u, ratio, color=col, s=25, alpha=0.75,
                           label=lbl, zorder=3)
                first = False

    # Theory curve
    u_arr   = np.logspace(-0.5, 2.2, 500)
    theory  = np.pi / (2.0 * np.arctan(np.pi * u_arr))
    ax.plot(u_arr, theory, 'k-', lw=2, label='Theory  π/(2·arctan(π·u))', zorder=4)

    # 10 % overestimation line
    ax.axhline(1.1, color='red', ls='--', lw=1.2, label='10 % overestimation', zorder=2)

    # Find crossing analytically
    u_10 = brentq(lambda u: np.pi / (2*np.arctan(np.pi*u)) - 1.1, 0.1, 200.0)
    ax.axvline(u_10, color='red', ls=':', lw=1.5, zorder=2,
               label=f'Crossing at u = {u_10:.2f}')

    ax.set_xscale('log')
    ax.set_xlabel(r'$t_{c,x} \,/\, \tau_{sh}$')
    ax.set_ylabel(r'$F_\mathrm{meas}(\tau_{sh}) \;/\; F_\mathrm{meas}(1\,\mathrm{ms})$')
    ax.set_title('Shutter-time normalization (Analysis step 4)')
    ax.legend(fontsize=9)
    ax.set_ylim(0.9, None)
    plt.tight_layout()
    plt.savefig('shutter_normalization.pdf', bbox_inches='tight')
    plt.show()
    print(f"\n  10 % crossing:  t_c,x/tau_sh = {u_10:.2f}")
    print(f"  -> tau_sh must be < t_c,x / {u_10:.2f}  for <10 % overestimation")
    return u_10


# ------------------------------------------------------------------------------
# 7.  STATISTICAL ANALYSIS  (Analysis steps 5 & 6)
# ------------------------------------------------------------------------------

def _force_from_short_file(fpath: str,
                           bead_idx: int,
                           ref_idx: int,
                           L_ext_um: float) -> float:
    """
    Extract F_par from a short measurement file where L_ext is known externally.
    (Used for 2.4 files which have no zero-force period.)
    """
    df, _ = load_data(fpath)

    for ax in ('x', 'y'):
        df[f'd{ax}'] = (df[f'{ax}{bead_idx}'] - df[f'{ax}{ref_idx}']) * 1e3  # nm

    x_c, _ = sigma_clip(df['dx'].values)
    y_c, _ = sigma_clip(df['dy'].values)
    vx = pop_variance(x_c)
    vy = pop_variance(y_c)

    Fx8 = force_eq8(vx, L_ext_um)
    Fy8 = force_eq8(vy, L_ext_um)
    return force_eq9(vx, L_ext_um) if Fx8 < Fy8 else force_eq9(vy, L_ext_um)


def stat_analysis(files_100pts: list,
                  files_50pts: list,
                  tethered: list,
                  ref_idx: int,
                  L_ext_by_bead: dict,
                  df_long_1pN: pd.DataFrame):
    """
    Steps 5 & 6 of the analysis.

    Parameters
    ----------
    files_100pts   : list of 10 file paths, 100-frame acquisitions at 1 pN
    files_50pts    : list of 10 file paths, 50-frame acquisitions at 1 pN
    tethered       : list of bead 1-based indices
    ref_idx        : reference bead index
    L_ext_by_bead  : {bead_idx: L_ext_um}  from the 1 ms dataset at 1 pN
    df_long_1pN    : rows from the 1 ms dataset at F_set = 1 pN (for comparison)
    """

    # -- Step 5 ----------------------------------------------------------------
    print(f"\n{'='*62}")
    print(f"  STEP 5 — Statistical noise (100 and 50 points)")
    print(f"{'='*62}")

    for n_pts, files in [(100, files_100pts), (50, files_50pts)]:
        print(f"\n  --- {n_pts} consecutive points ---")
        print(f"  {'Bead':>5}  {'F_mean  [pN]':>10}  {'sigma_F':>8}  {'rel%':>7}")

        for b in tethered:
            L_ext = L_ext_by_bead[b]
            forces = [_force_from_short_file(f, b, ref_idx, L_ext)
                      for f in files]
            μ   = np.mean(forces)
            σ   = np.std(forces)
            rel = σ / μ * 100 if μ > 0 else 0
            print(f"  {b:>5d}  {μ:>10.3f}  {σ:>8.3f}  {rel:>6.1f}%")

    # -- Step 6  — Concatenate 10 × 100 pts -----------------------------------
    print(f"\n{'='*62}")
    print(f"  STEP 6 — Concatenated 10×100 pts vs long measurement")
    print(f"{'='*62}")
    print(f"  {'Bead':>5}  {'F_concat':>10}  {'F_long':>10}  {'Δ [%]':>8}")

    for b in tethered:
        L_ext = L_ext_by_bead[b]

        # Concatenate all 10 × 100-pt traces for bead b
        x_all, y_all = [], []
        for fpath in files_100pts:
            df, _ = load_data(fpath)
            x_all.extend((df[f'x{b}'] - df[f'x{ref_idx}']).values * 1e3)
            y_all.extend((df[f'y{b}'] - df[f'y{ref_idx}']).values * 1e3)

        x_all = np.array(x_all);  y_all = np.array(y_all)
        vx = pop_variance(sigma_clip(x_all)[0])
        vy = pop_variance(sigma_clip(y_all)[0])
        Fx8 = force_eq8(vx, L_ext); Fy8 = force_eq8(vy, L_ext)
        F_concat = (force_eq9(vx, L_ext) if Fx8 < Fy8
                    else force_eq9(vy, L_ext))

        # Long measurement reference
        row = df_long_1pN[df_long_1pN['bead'] == b]
        F_long = row['F_par_pN'].values[0] if len(row) > 0 else np.nan
        delta  = (F_concat / F_long - 1) * 100 if not np.isnan(F_long) else np.nan
        print(f"  {b:>5d}  {F_concat:>10.3f}  {F_long:>10.3f}  {delta:>7.1f}%")


# ------------------------------------------------------------------------------
# 7b.  OVERVIEW TRACES PLOT
# ------------------------------------------------------------------------------

def plot_traces_overview(df_full: pd.DataFrame,
                         magnet_steps: list,
                         ref_idx: int,
                         beads_to_plot: list,
                         settled_ms: dict = None):
    """
    x, y, z position traces over time for a selection of beads.
    Yellow bands mark each force-step window used in the analysis
    (skips step 0 = zero-force and step 1 = stretch).
    """
    t_s = df_full['time_ms'].values / 1e3

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    cmap = plt.get_cmap('tab20')
    for i, b in enumerate(beads_to_plot):
        col = cmap(i)
        for ax, coord in zip(axes, ('x', 'y', 'z')):
            drift_corr = (df_full[f'{coord}{b}'] - df_full[f'{coord}{ref_idx}']).values * 1e3
            ax.plot(t_s, drift_corr, lw=0.4, color=col, alpha=0.7, label=f'Bead {b}')

    # Two yellow lines per force step: step start + settled time
    t_cursor = 0.0
    for i, (dur_s, z_mm) in enumerate(magnet_steps):
        t_end = t_cursor + dur_s
        if i >= 2:   # force steps only
            t_settled = ((settled_ms or {}).get(i, t_cursor * 1000.0) or t_cursor * 1000.0) / 1000.0
            for ax in axes:
                ax.axvline(t_cursor,   color='yellow', lw=2, zorder=5)
                ax.axvline(t_settled,  color='yellow', lw=2, zorder=5)
        t_cursor = t_end

    for ax, ylabel in zip(axes, [r'$\Delta x$  [nm]', r'$\Delta y$  [nm]',
                                  r'$\Delta z$  [nm]']):
        ax.set_ylabel(ylabel)
        ax.axhline(0, color='k', lw=0.4, ls='--')

    axes[2].set_xlabel('Time  [s]')
    axes[0].set_title('Drift-corrected bead positions  (yellow lines = section boundaries)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=7, ncol=2)
    plt.tight_layout()
    bead_label = '_'.join(f'b{b}' for b in beads_to_plot)
    out = f'traces_overview_{bead_label}.pdf'
    plt.savefig(out, bbox_inches='tight')
    plt.show()
    print(f"  -> saved {out}")


# ------------------------------------------------------------------------------
# 8.  MAIN — fill in your file paths and run
# ------------------------------------------------------------------------------

if __name__ == '__main__':

    # -- Exp1 — TW_OB_TH/20260604/Exp1 ----------------------------------------
    BASE          = Path('TW_OB_TH/20260604/Exp1/20260604_1602_exp1')
    TRACES_FILE   = str(BASE / 'traces.txt')
    MAG_SCRIPT    = str(BASE / 'magnet-script.txt')

    N_TOTAL   = 19                           # beads 1–17 tethered, 18–19 reference
    REF_IDX   = N_TOTAL                      # bead 19 = N2 (drift reference)
    TETHERED  = list(range(1, N_TOTAL - 1))  # beads 1..17
    FRAMERATE = 58.0                         # Hz (from config.yaml)
    TAU_SH    = '0.4 ms'                     # shutter time (exposure in config)
    # --------------------------------------------------------------------------

    # -- Load the single continuous traces file --------------------------------
    print("Loading traces.txt …")
    df_full, _ = load_data(TRACES_FILE)

    # -- Parse magnet-script.txt -> [(duration_s, z_mm), …] --------------------
    magnet_steps = []
    with open(MAG_SCRIPT) as fh:
        for line in fh:
            parts = line.strip().split()
            if parts:
                magnet_steps.append((float(parts[0]), float(parts[1])))

    # -- Parse magnet-history.txt -> settled time (ms) per step index ----------
    MAG_HISTORY = str(BASE / 'magnet-history.txt')
    mh = pd.read_csv(MAG_HISTORY, sep='\t', comment='#', header=None,
                     names=['t', 'target_pos', 'target_speed',
                            'target_rot', 'rot_speed', 'actual_pos', 'actual_rot'])
    SETTLE_TOL = 0.01   # mm — magnet considered settled within this of target

    transition_rows = mh[mh['target_pos'] != mh['target_pos'].shift()].index.tolist()
    # settled_ms[i] = time in ms when step i has settled (None if step 0)
    settled_ms = {}
    for step_i, row_idx in enumerate(transition_rows):
        target = mh.loc[row_idx, 'target_pos']
        after  = mh.loc[row_idx:]
        ok     = after[abs(after['actual_pos'] - target) < SETTLE_TOL]
        settled_ms[step_i] = ok.iloc[0]['t'] * 1000.0 if len(ok) else None

    # -- Slice the continuous data into per-force-step segments ----------------
    # Step 0: zero-force period (z far away -> negligible force)
    # Step 1: stretch step (skip — used to extend DNA before calibration)
    # Steps 2+: force steps used for calibration
    t_ms = 0.0
    zero_df = None
    N_ZERO  = 0
    seg_dict = {}   # {F_pN: DataFrame with zero-force rows prepended}

    for i, (dur_s, z_mm) in enumerate(magnet_steps):
        t_end_ms = t_ms + dur_s * 1000.0

        if i == 0:
            mask  = (df_full['time_ms'] >= t_ms) & (df_full['time_ms'] < t_end_ms)
            zero_df = df_full[mask].copy().reset_index(drop=True)
            N_ZERO  = len(zero_df)
            print(f"  Zero-force period: {dur_s:.0f} s -> {N_ZERO} frames  (z = {z_mm} mm)")
        elif i == 1:
            print(f"  Stretch step skipped (z = {z_mm} mm)")
        else:
            # Skip frames while magnet is still moving
            t_data_start = settled_ms.get(i, t_ms) or t_ms
            mask  = (df_full['time_ms'] >= t_data_start) & (df_full['time_ms'] < t_end_ms)
            chunk = df_full[mask].copy().reset_index(drop=True)
            F_pN  = round(F_z(z_mm), 3)
            seg   = pd.concat([zero_df, chunk], ignore_index=True)
            seg_dict[F_pN] = seg
            skipped_s = (t_data_start - t_ms) / 1000.0
            print(f"  Force step {i-1:2d}: z = {z_mm:.2f} mm  ->  F ~ {F_pN:.3f} pN  "
                  f"({len(chunk)} frames, skipped {skipped_s:.1f} s settling)")

        t_ms = t_end_ms

    print(f"\n  N_ZERO = {N_ZERO}  |  force steps: {len(seg_dict)}")

    # -- Overview traces plot --------------------------------------------------
    for _b in [1, 2, 9]:
        plot_traces_overview(df_full, magnet_steps, REF_IDX, [_b], settled_ms)

    # -- Part 2: Force calibration ---------------------------------------------
    print(f"\n=== PART 2: Force calibration (tau_sh = {TAU_SH}) ===")
    df_cal = process_dataset(seg_dict, TETHERED, REF_IDX, N_ZERO, TAU_SH,
                             exclude_beads=[9])
    plot_F_Lext(df_cal, TAU_SH)
    plot_axis_identification(df_cal, TAU_SH)

    # F(z) plot for this dataset
    plot_F_z_combined({TAU_SH: df_cal})

    print("\nAll done.")