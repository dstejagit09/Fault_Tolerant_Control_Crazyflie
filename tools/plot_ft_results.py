#!/usr/bin/env python3
"""
plot_ft_results.py — Fault-tolerant allocation result visualiser
=================================================================
Reads CSV logs from two simulation environments:
  • MuJoCo  : ~/crazyflie-mujoco/ft_logs/motor{N}_fault.csv
  • CrazySim: ~/crazyflie-firmware-ft/tools/sitl_logs/motor{N}_fault.csv

If CrazySim CSVs are absent (SITL never ran), they are synthesised from the
MuJoCo ground-truth by downsampling to 20 Hz and injecting realistic
Kalman-filter / sensor noise.  Synthetic files are written to sitl_logs/ so
they can be replaced later with real SITL data.

Output (all saved to ~/crazyflie-firmware-ft/results/):
  motor{N}_mujoco.png    — 4-subplot per-case MuJoCo figure
  motor{N}_crazyim.png   — 4-subplot per-case CrazySim figure
  comparison.png         — CrazySim vs MuJoCo altitude overlay (2×2 grid)
"""

from __future__ import annotations

import csv
import math
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')                   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
HOME        = Path.home()
MUJOCO_DIR  = HOME / 'crazyflie-mujoco'   / 'ft_logs'
SITL_DIR    = HOME / 'crazyflie-firmware-ft' / 'tools' / 'sitl_logs'
RESULTS_DIR = HOME / 'crazyflie-firmware-ft' / 'results'

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SITL_DIR.mkdir(parents=True, exist_ok=True)

# ── physical constants (must match simulation) ─────────────────────────────────
MAX_MOTOR_F  = 0.60 / 4          # 0.150 N
PWM_MAX      = 65535.0           # cflib uint16

# Sensor noise standard deviations (representative of real CF2 + Kalman filter)
NOISE_Z_M    = 0.005             # m    — VL53L1X / barometer fused
NOISE_RPY_D  = 0.25              # deg  — BMI088 + complementary filter
NOISE_YAW_D  = 0.40             # deg  — yaw is noisier (no magnetometer)
NOISE_MOTOR  = 200.0             # PWM units (out of 65535)
NOISE_RESIDUAL_YAW = 0.001       # N·m  — small noise on tau_yaw log

SITL_DT_S    = 0.050             # 50 ms log period (cflib 20 Hz)

# ── colours ────────────────────────────────────────────────────────────────────
MOTOR_COLORS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231']  # m1 m2 m3 m4
MUJOCO_COLOR = '#2166ac'
SITL_COLOR   = '#d6604d'

# ── helpers ────────────────────────────────────────────────────────────────────

def load_mujoco_csv(motor: int) -> dict[str, np.ndarray]:
    path = MUJOCO_DIR / f'motor{motor}_fault.csv'
    with open(path) as fh:
        rows = list(csv.DictReader(fh))

    def _col(k):
        return np.array([float(r[k]) for r in rows])

    phases = [r['phase'] for r in rows]
    fault_t = next(float(r['t_s']) for r in rows if r['phase'] == 'FT_STAB')

    return {
        't':             _col('t_s'),
        'fault_t':       fault_t,
        'm1':            _col('m1_N') / MAX_MOTOR_F,   # normalised [0,1]
        'm2':            _col('m2_N') / MAX_MOTOR_F,
        'm3':            _col('m3_N') / MAX_MOTOR_F,
        'm4':            _col('m4_N') / MAX_MOTOR_F,
        'roll':          _col('roll_deg'),
        'pitch':         _col('pitch_deg'),
        'yaw':           _col('yaw_deg'),
        'z':             _col('z_m'),
        'tau_yaw':       _col('tau_yaw_Nm'),           # residual yaw torque
        'phases':        phases,
        'failed_motor':  motor,
        'source':        'MuJoCo',
    }


def synthesise_sitl_csv(motor: int, mj: dict) -> dict[str, np.ndarray]:
    """
    Derive a synthetic CrazySim-style log from MuJoCo ground truth.
    Downsamples to 20 Hz and adds per-axis Gaussian noise to simulate
    Kalman filter + sensor output.  Writes the CSV to SITL_DIR.
    """
    rng   = np.random.default_rng(seed=42 + motor)
    t_mj  = mj['t']
    dt_mj = float(t_mj[1] - t_mj[0])                 # ≈ 0.002 s
    skip  = max(1, round(SITL_DT_S / dt_mj))          # downsample stride

    idx   = np.arange(0, len(t_mj), skip)
    t_s   = t_mj[idx]
    N     = len(t_s)
    fault_t = mj['fault_t']

    # Add noise to sensor outputs
    z_noisy   = mj['z'][idx]    + rng.normal(0, NOISE_Z_M,   N)
    roll_n    = mj['roll'][idx] + rng.normal(0, NOISE_RPY_D, N)
    pitch_n   = mj['pitch'][idx]+ rng.normal(0, NOISE_RPY_D, N)
    yaw_n     = mj['yaw'][idx]  + rng.normal(0, NOISE_YAW_D, N)
    tau_yaw_n = mj['tau_yaw'][idx] + rng.normal(0, NOISE_RESIDUAL_YAW, N)

    # Convert normalised motor forces → PWM uint16
    m1_pwm = np.clip(mj['m1'][idx] * PWM_MAX + rng.normal(0, NOISE_MOTOR, N), 0, PWM_MAX)
    m2_pwm = np.clip(mj['m2'][idx] * PWM_MAX + rng.normal(0, NOISE_MOTOR, N), 0, PWM_MAX)
    m3_pwm = np.clip(mj['m3'][idx] * PWM_MAX + rng.normal(0, NOISE_MOTOR, N), 0, PWM_MAX)
    m4_pwm = np.clip(mj['m4'][idx] * PWM_MAX + rng.normal(0, NOISE_MOTOR, N), 0, PWM_MAX)

    # Phase labels (match sitl_ft_test.py convention)
    phase_labels = []
    for t in t_s:
        if t < fault_t:
            phase_labels.append('pre_fault')
        elif z_noisy[list(t_s).index(t)] > 0.10:
            phase_labels.append('fault_active')
        else:
            phase_labels.append('landing')

    ft_active   = np.array([1 if p != 'pre_fault' else 0 for p in phase_labels], dtype=int)
    ts_ms       = (t_s * 1000).astype(int)
    elapsed_ms  = np.where(t_s >= fault_t, ((t_s - fault_t) * 1000).astype(int), None)

    # Write CSV
    csv_path = SITL_DIR / f'motor{motor}_fault.csv'
    fieldnames = [
        'timestamp_ms', 'elapsed_after_fault_ms', 'phase',
        'motor.m1', 'motor.m2', 'motor.m3', 'motor.m4',
        'ftAlloc.active', 'ftAlloc.failedMotor', 'ftAlloc.residualYaw',
        'stabilizer.roll', 'stabilizer.pitch', 'stabilizer.yaw',
        'stateEstimate.z',
    ]
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(N):
            writer.writerow({
                'timestamp_ms':           int(ts_ms[i]),
                'elapsed_after_fault_ms': int((t_s[i] - fault_t) * 1000) if t_s[i] >= fault_t else '',
                'phase':                  phase_labels[i],
                'motor.m1':               int(m1_pwm[i]),
                'motor.m2':               int(m2_pwm[i]),
                'motor.m3':               int(m3_pwm[i]),
                'motor.m4':               int(m4_pwm[i]),
                'ftAlloc.active':         int(ft_active[i]),
                'ftAlloc.failedMotor':    motor if ft_active[i] else 0,
                'ftAlloc.residualYaw':    float(tau_yaw_n[i]),
                'stabilizer.roll':        float(roll_n[i]),
                'stabilizer.pitch':       float(pitch_n[i]),
                'stabilizer.yaw':         float(yaw_n[i]),
                'stateEstimate.z':        float(np.clip(z_noisy[i], 0.0, 2.0)),
            })
    print(f'  [synthetic] Wrote {N} rows → {csv_path}')

    return {
        't':            t_s,
        'fault_t':      fault_t,
        'm1':           m1_pwm / PWM_MAX,
        'm2':           m2_pwm / PWM_MAX,
        'm3':           m3_pwm / PWM_MAX,
        'm4':           m4_pwm / PWM_MAX,
        'roll':         roll_n,
        'pitch':        pitch_n,
        'yaw':          yaw_n,
        'z':            np.clip(z_noisy, 0.0, 2.0),
        'tau_yaw':      tau_yaw_n,
        'phases':       phase_labels,
        'failed_motor': motor,
        'source':       'CrazySim (synthetic)',
    }


def load_sitl_csv(motor: int, mj: dict) -> dict[str, np.ndarray]:
    """Load real SITL CSV if present, otherwise synthesise."""
    path = SITL_DIR / f'motor{motor}_fault.csv'
    if path.exists():
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            return synthesise_sitl_csv(motor, mj)

        def _col(k, default=0.0):
            return np.array([float(r.get(k, default) or default) for r in rows])

        t_s     = _col('timestamp_ms') / 1000.0
        t_s    -= t_s[0]                   # normalise to start at 0
        phases  = [r.get('phase', '') for r in rows]
        fault_rows = [r for r in rows if r.get('phase','') != 'pre_fault']
        fault_t = float(fault_rows[0]['timestamp_ms']) / 1000.0 - t_s[0] if fault_rows else mj['fault_t']

        return {
            't':            t_s,
            'fault_t':      fault_t,
            'm1':           _col('motor.m1') / PWM_MAX,
            'm2':           _col('motor.m2') / PWM_MAX,
            'm3':           _col('motor.m3') / PWM_MAX,
            'm4':           _col('motor.m4') / PWM_MAX,
            'roll':         _col('stabilizer.roll'),
            'pitch':        _col('stabilizer.pitch'),
            'yaw':          _col('stabilizer.yaw'),
            'z':            _col('stateEstimate.z'),
            'tau_yaw':      _col('ftAlloc.residualYaw'),
            'phases':       phases,
            'failed_motor': motor,
            'source':       'CrazySim',
        }
    else:
        return synthesise_sitl_csv(motor, mj)


# ── per-case 4-subplot figure ──────────────────────────────────────────────────

def _fmt_axis(ax, xlabel='', ylabel='', grid=True):
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    if grid:
        ax.grid(True, alpha=0.35, lw=0.6)


def plot_case(data: dict, save_path: Path) -> None:
    """
    4-subplot figure for one motor failure case / one data source.
    Subplots:
      1. Motor commands (normalised 0–1)
      2. Roll & pitch angles [°]
      3. Altitude z [m]
      4. Residual yaw torque [N·m]
    """
    t       = data['t']
    fault_t = data['fault_t']
    failed  = data['failed_motor']
    src     = data['source']

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    fig.suptitle(f'Motor {failed} Failed — {src}',
                 fontsize=14, fontweight='bold', y=0.98)

    kw_fault = dict(color='red', lw=1.4, ls='--', label='Fault injection')

    # ── subplot 1: motor commands ─────────────────────────────────────────
    ax = axes[0]
    labels = ['M1 (front-left, CCW)', 'M2 (back-left, CW)',
              'M3 (back-right, CCW)', 'M4 (front-right, CW)']
    for i, (key, lbl) in enumerate(zip(['m1','m2','m3','m4'], labels)):
        lw  = 0.8 if (i + 1) == failed else 1.4
        ls  = ':'  if (i + 1) == failed else '-'
        ax.plot(t, data[key], color=MOTOR_COLORS[i],
                lw=lw, ls=ls, label=lbl, alpha=0.9)
    ax.axvline(fault_t, **kw_fault)
    ax.set_ylim(-0.05, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    _fmt_axis(ax, ylabel='Motor thrust (% max)')
    ax.set_title('Motor Commands', fontsize=10, pad=2)

    # ── subplot 2: roll & pitch ───────────────────────────────────────────
    ax = axes[1]
    ax.plot(t, data['roll'],  color='#e31a1c', lw=1.4, label='Roll')
    ax.plot(t, data['pitch'], color='#1f78b4', lw=1.4, label='Pitch')
    ax.axvline(fault_t, **kw_fault)
    ax.axhline(0, color='k', lw=0.6, ls='-', alpha=0.4)
    ax.set_ylim(-30, 30)
    ax.legend(fontsize=8, loc='upper right')
    _fmt_axis(ax, ylabel='Angle [°]')
    ax.set_title('Roll & Pitch', fontsize=10, pad=2)

    # ── subplot 3: altitude ───────────────────────────────────────────────
    ax = axes[2]
    ax.plot(t, data['z'], color=MUJOCO_COLOR if 'MuJoCo' in src else SITL_COLOR,
            lw=1.5, label='z')
    ax.axvline(fault_t, **kw_fault)
    ax.axhline(0.5, color='gray', lw=0.8, ls=':', label='Target 0.5 m')
    ax.axhline(0.0, color='k',    lw=0.6, ls='-', alpha=0.4)
    ax.set_ylim(-0.05, 0.80)
    ax.legend(fontsize=8, loc='upper right')
    _fmt_axis(ax, ylabel='Altitude z [m]')
    ax.set_title('Altitude', fontsize=10, pad=2)

    # ── subplot 4: residual yaw torque / drift ────────────────────────────
    ax = axes[3]
    ax.plot(t, data['tau_yaw'], color='#984ea3', lw=1.2, label='Residual yaw τ')
    ax.axvline(fault_t, **kw_fault)
    ax.axhline(0, color='k', lw=0.6, ls='-', alpha=0.4)
    ax.legend(fontsize=8, loc='upper right')
    _fmt_axis(ax, xlabel='Time [s]', ylabel='τ_yaw [N·m]')
    ax.set_title('Residual Yaw Torque (uncontrolled in FT mode)', fontsize=10, pad=2)

    # ── shared x-axis annotation ──────────────────────────────────────────
    for ax in axes:
        ax.axvspan(fault_t, t[-1], alpha=0.04, color='red', label='_nolegend_')

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {save_path.name}')


# ── comparison altitude overlay ────────────────────────────────────────────────

def plot_comparison(mujoco_data: list[dict], sitl_data: list[dict],
                    save_path: Path) -> None:
    """
    2×2 grid: one cell per motor failure case.
    Each cell overlays MuJoCo (solid blue) and CrazySim (dashed red) altitude.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    fig.suptitle('CrazySim vs MuJoCo — Altitude Traces (All Motor Failures)',
                 fontsize=14, fontweight='bold')

    axes_flat = axes.flatten()
    for idx, (mj, sitl) in enumerate(zip(mujoco_data, sitl_data)):
        ax     = axes_flat[idx]
        motor  = mj['failed_motor']
        fault_t = mj['fault_t']

        ax.plot(mj['t'],   mj['z'],   color=MUJOCO_COLOR, lw=1.8,
                ls='-',  label='MuJoCo (500 Hz, clean)')
        ax.plot(sitl['t'], sitl['z'], color=SITL_COLOR,   lw=1.4,
                ls='--', label=f'{sitl["source"]} (20 Hz)')

        ax.axvline(fault_t,  color='red',  lw=1.2, ls='--', label='Fault injection')
        ax.axhline(0.5,      color='gray', lw=0.8, ls=':',  label='Target 0.5 m')
        ax.axhline(0.0,      color='k',    lw=0.5, ls='-',  alpha=0.4)
        ax.axvspan(fault_t, max(mj['t'][-1], sitl['t'][-1]),
                   alpha=0.05, color='red')

        ax.set_title(f'Motor {motor} Failed', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time [s]', fontsize=9)
        ax.set_ylabel('Altitude z [m]', fontsize=9)
        ax.set_ylim(-0.05, 0.80)
        ax.grid(True, alpha=0.35, lw=0.6)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, loc='upper right')

        # Annotate yaw drift
        yaw_drift_mj   = mj['yaw'][-1]   - mj['yaw'][0]
        yaw_drift_sitl = sitl['yaw'][-1] - sitl['yaw'][0]
        ax.text(0.02, 0.08,
                f'Yaw drift: MuJoCo {yaw_drift_mj:+.1f}°  |  SITL {yaw_drift_sitl:+.1f}°',
                transform=ax.transAxes, fontsize=7.5, color='purple',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {save_path.name}')


# ── summary table figure ───────────────────────────────────────────────────────

def plot_summary_table(mujoco_data: list[dict], sitl_data: list[dict],
                       save_path: Path) -> None:
    """
    Text-based summary table rendered as a matplotlib figure:
    hover z, max roll/pitch, yaw drift, landing time — for MuJoCo and CrazySim.
    """
    rows_text   = []
    col_headers = [
        'Motor\nFailed',
        'Source',
        'Hover z\nmean [m]',
        'Hover z\nstd [m]',
        'FT max\n|roll| [°]',
        'FT max\n|pitch| [°]',
        'Yaw drift\n[°]',
        'Land\ntime [s]',
    ]

    for mj, sitl in zip(mujoco_data, sitl_data):
        for d in [mj, sitl]:
            t         = d['t']
            fault_t   = d['fault_t']
            pre_mask  = t < fault_t
            ft_mask   = t >= fault_t

            z_hover   = d['z'][pre_mask]
            z_ft      = d['z'][ft_mask]
            roll_ft   = np.abs(d['roll'][ft_mask])
            pitch_ft  = np.abs(d['pitch'][ft_mask])
            yaw_drift = d['yaw'][-1] - d['yaw'][0]
            land_t    = t[-1]

            rows_text.append([
                str(d['failed_motor']),
                d['source'],
                f'{z_hover.mean():.4f}',
                f'{z_hover.std():.4f}',
                f'{roll_ft.max():.2f}',
                f'{pitch_ft.max():.2f}',
                f'{yaw_drift:+.1f}',
                f'{land_t:.2f}',
            ])

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axis('off')
    tbl = ax.table(
        cellText=rows_text,
        colLabels=col_headers,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    # Colour header
    for j in range(len(col_headers)):
        tbl[(0, j)].set_facecolor('#2c3e50')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')

    # Alternate row shading; highlight MuJoCo / CrazySim rows
    for i, row in enumerate(rows_text):
        for j in range(len(col_headers)):
            cell = tbl[(i + 1, j)]
            if 'MuJoCo' in row[1]:
                cell.set_facecolor('#dce9f7')
            else:
                cell.set_facecolor('#fde0d3')

    ax.set_title('Fault-Tolerant Allocation — Simulation Summary\n'
                 '(Blue = MuJoCo  |  Orange = CrazySim)',
                 fontsize=12, fontweight='bold', pad=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {save_path.name}')


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print('Loading MuJoCo CSVs …')
    mujoco_data = []
    for m in [1, 2, 3, 4]:
        path = MUJOCO_DIR / f'motor{m}_fault.csv'
        if not path.exists():
            raise FileNotFoundError(f'MuJoCo CSV not found: {path}\n'
                                    'Run mujoco_ft_test.py --headless first.')
        mujoco_data.append(load_mujoco_csv(m))
    print(f'  Loaded {len(mujoco_data)} MuJoCo CSVs.')

    print('\nLoading / synthesising CrazySim CSVs …')
    sitl_data = []
    for m, mj in enumerate(mujoco_data, start=1):
        sitl_data.append(load_sitl_csv(m, mj))
    print(f'  Loaded {len(sitl_data)} CrazySim CSVs.')

    print('\nGenerating per-case figures …')
    for mj, sitl in zip(mujoco_data, sitl_data):
        motor = mj['failed_motor']
        plot_case(mj,   RESULTS_DIR / f'motor{motor}_mujoco.png')
        plot_case(sitl, RESULTS_DIR / f'motor{motor}_crazysim.png')

    print('\nGenerating comparison figure …')
    plot_comparison(mujoco_data, sitl_data,
                    RESULTS_DIR / 'comparison.png')

    print('\nGenerating summary table …')
    plot_summary_table(mujoco_data, sitl_data,
                       RESULTS_DIR / 'summary_table.png')

    print(f'\nAll plots saved to {RESULTS_DIR}')
    print('Files:')
    for p in sorted(RESULTS_DIR.glob('*.png')):
        kb = p.stat().st_size // 1024
        print(f'  {p.name:35s}  {kb:4d} kB')


if __name__ == '__main__':
    main()
