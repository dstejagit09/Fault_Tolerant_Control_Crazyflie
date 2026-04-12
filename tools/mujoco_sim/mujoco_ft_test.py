#!/usr/bin/env python3
"""
mujoco_ft_test.py — Crazyflie MuJoCo Fault-Injection Test
==========================================================
Independent validation of the fault-tolerant 3-motor closed-form
allocation formulas from power_distribution_quadrotor.c.

Flight sequence (per test case):
  0 – FAULT_TIME s  : 4-motor hover at 0.5 m
  FAULT_TIME s      : motor N killed → switch to FT 3-motor allocator
  +2 s              : FT stabilisation hold at 0.5 m
  descent           : target_z decremented at DESCENT_RATE m/s until landed
  landed            : z < LAND_Z, stop setpoint

All 4 failure cases are run sequentially.  If a viewer is requested, all
cases play in the *same* window (model is reset between cases so you can
watch each landing in sequence).

Logged at every timestep (0.002 s):
  t, phase, m1_N, m2_N, m3_N, m4_N,
  roll_deg, pitch_deg, yaw_deg, x_m, y_m, z_m

One CSV file is written per case:
  <output_dir>/motor<N>_fault.csv

Usage
-----
# All 4 cases with viewer (press Esc between cases or let each finish):
python3 mujoco_ft_test.py

# Headless batch (fast):
python3 mujoco_ft_test.py --headless

# Single case with viewer:
python3 mujoco_ft_test.py --cases 2

# Custom output dir:
python3 mujoco_ft_test.py --headless --output-dir /tmp/ft_logs

Allocation formulas (firmware variables: r = roll/2, p = pitch/2)
------------------------------------------------------------------
  Motor 1 fails → m2=2(T-r),  m3=2(r-p),  m4=2(T+p)
  Motor 2 fails → m1=2(T-r),  m3=2(T-p),  m4=2(r+p)
  Motor 3 fails → m1=2(p-r),  m2=2(T-p),  m4=2(T+r)
  Motor 4 fails → m1=2(T+p),  m2=-2(r+p), m3=2(T+r)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np

# ── default paths ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent
_FW_TOOLS   = Path.home() / 'crazyflie-firmware-ft' / 'tools' / 'mujoco_sim'
DEFAULT_XML = _FW_TOOLS / 'cf2_sim.xml'
DEFAULT_OUT = _SCRIPT_DIR / 'ft_logs'

# ── physical constants (CF2 datasheet + MIT system-ID Landry 2015) ─────────────
MASS           = 0.027          # kg
G              = 9.81           # m/s²
HOVER_F        = MASS * G       # ≈ 0.2648 N total thrust at hover

ARM_LEN        = 0.046          # m  (motor centre to CF2 centre, along diagonal)
ARM            = ARM_LEN * math.sin(math.radians(45))   # ≈ 0.03253 m moment arm

THRUST2TORQUE  = 0.005964552    # N·m / N  (reaction torque ratio)

# Per-motor max: CF2.1 hovers at ~44% of max → max ≈ 0.60/4 = 0.150 N
# FT allocator needs 2×T_hover ≈ 0.132 N per motor → within headroom.
MAX_MOTOR_F    = 0.60 / 4       # 0.150 N

# Spin directions for yaw reaction torque (CCW=+1, CW=−1)
#   M1 front-left  CCW (+), M2 back-left  CW (−),
#   M3 back-right  CCW (+), M4 front-right CW (−)
SPIN = np.array([1.0, -1.0, 1.0, -1.0])

# Inertia (used for attitude PD gain sizing)
Ixx = 2.3951e-5   # kg·m²
Izz = 3.2347e-5   # kg·m²

# ── flight parameters ──────────────────────────────────────────────────────────
HOVER_Z       = 0.5     # m  — target hover altitude
FAULT_TIME    = 3.0     # s  — fault injection time
FT_STAB_TIME  = 2.0     # s  — post-fault stabilisation hold before descent
DESCENT_RATE  = 0.10    # m/s — controlled descent rate
LAND_Z        = 0.06    # m  — below this → consider landed
TOTAL_TIMEOUT = 25.0    # s  — maximum simulation time per case


# ── flight phases ──────────────────────────────────────────────────────────────
class Phase(Enum):
    HOVER    = auto()   # 4-motor, holding HOVER_Z
    FT_STAB  = auto()   # 3-motor FT, stabilising after fault
    DESCENT  = auto()   # 3-motor FT, descending
    LANDED   = auto()   # on the ground


# ── propeller colours ──────────────────────────────────────────────────────────
RGBA_OK   = np.array([0.20, 0.80, 0.20, 1.0])   # green  — healthy
RGBA_FAIL = np.array([0.90, 0.10, 0.10, 1.0])   # red    — dead motor


# ── mixer / allocation ─────────────────────────────────────────────────────────

def alloc_4motor(T: float, r: float, p: float, y: float) -> np.ndarray:
    """
    Standard Crazyflie mixer in physical (per-motor Newton) units.
      T = F_total / 4
      r = tau_roll  / (4 * ARM)
      p = tau_pitch / (4 * ARM)
      y = tau_yaw   / (4 * THRUST2TORQUE)
    """
    return np.clip([T-r+p+y, T-r-p-y, T+r-p+y, T+r+p-y], 0.0, MAX_MOTOR_F)


def alloc_fault_tolerant(T: float, r: float, p: float,
                         failed: int) -> tuple[np.ndarray, float]:
    """
    Closed-form 3-motor allocation (power_distribution_quadrotor.c).
    Prioritises exact T, r, p tracking; yaw is uncontrolled residual.

    Returns
    -------
    forces       : ndarray shape (4,), per-motor thrust [N], failed motor = 0
    yaw_residual : float, uncontrolled yaw torque [N·m]
    """
    m = np.zeros(4)

    if failed == 1:
        m[1] = 2.0 * (T - r)
        m[2] = 2.0 * (r - p)
        m[3] = 2.0 * (T + p)
        yaw_N = -T + r - p

    elif failed == 2:
        m[0] = 2.0 * (T - r)
        m[2] = 2.0 * (T - p)
        m[3] = 2.0 * (r + p)
        yaw_N = T - r - p

    elif failed == 3:
        m[0] = 2.0 * (p - r)
        m[1] = 2.0 * (T - p)
        m[3] = 2.0 * (T + r)
        yaw_N = p - r - T

    elif failed == 4:
        m[0] = 2.0 * (T + p)
        m[1] = -2.0 * (r + p)
        m[2] = 2.0 * (T + r)
        yaw_N = T + r + p

    else:
        # fallback — should not reach here, but be safe
        return alloc_4motor(T, r, p, 0.0), 0.0

    forces       = np.clip(m, 0.0, MAX_MOTOR_F)
    yaw_residual = THRUST2TORQUE * 4.0 * yaw_N   # scale to N·m
    return forces, yaw_residual


def motors_to_wrench(f: np.ndarray) -> tuple[float, float, float, float]:
    """
    Per-motor forces → body-frame wrench (thrust N, roll N·m, pitch N·m, yaw N·m).
    """
    thrust    = float(f.sum())
    tau_roll  = ARM * float((f[2] + f[3]) - (f[0] + f[1]))
    tau_pitch = ARM * float((f[0] + f[3]) - (f[1] + f[2]))
    tau_yaw   = THRUST2TORQUE * float(SPIN @ f)
    return thrust, tau_roll, tau_pitch, tau_yaw


# ── cascaded PD/PID hover controller ──────────────────────────────────────────

@dataclass
class HoverController:
    """
    Outer loop: altitude z → total thrust.
    Inner loop: roll/pitch/yaw angles → torques.
    All outputs in physical units (N and N·m).
    """
    target_z: float = HOVER_Z

    # Altitude PID  (ωn ≈ 5 rad/s, critically damped)
    kp_z: float  = field(default_factory=lambda: MASS * 25.0)
    ki_z: float  = field(default_factory=lambda: MASS * 4.0)
    kd_z: float  = field(default_factory=lambda: MASS * 10.0)

    # Attitude PD  (ωn ≈ 20 rad/s)
    kp_rp:  float = field(default_factory=lambda: Ixx * 400.0)
    kd_rp:  float = field(default_factory=lambda: Ixx * 40.0)
    kp_yaw: float = field(default_factory=lambda: Izz * 25.0)
    kd_yaw: float = field(default_factory=lambda: Izz * 10.0)

    # integrator state (reset between runs)
    _z_integral: float = field(default=0.0, init=False, repr=False)

    def reset(self) -> None:
        self._z_integral = 0.0

    def compute(self, state: dict, dt: float) -> tuple[float, float, float, float]:
        """
        Returns (T, r, p, yaw_comp) — all in per-motor Newton units:
          T        = F_total / 4
          r        = tau_roll  / (4 * ARM)
          p        = tau_pitch / (4 * ARM)
          yaw_comp = tau_yaw   / (4 * THRUST2TORQUE)
        """
        z                  = float(state['pos'][2])
        vz                 = float(state['vel'][2])
        roll, pitch, yaw   = state['rpy']
        ox, oy, oz         = state['gyro']

        # Altitude PID
        z_err              = self.target_z - z
        self._z_integral  += z_err * dt
        self._z_integral   = float(np.clip(self._z_integral, -0.5, 0.5))   # anti-windup
        F_total            = (HOVER_F
                              + self.kp_z * z_err
                              + self.ki_z * self._z_integral
                              - self.kd_z * vz)
        F_total            = float(np.clip(F_total, 0.0, MAX_MOTOR_F * 4.0))

        # Attitude PD (zero reference — hold level)
        tau_roll  = -self.kp_rp  * roll  - self.kd_rp  * ox
        tau_pitch = -self.kp_rp  * pitch - self.kd_rp  * oy
        tau_yaw   = -self.kp_yaw * yaw   - self.kd_yaw * oz

        # Convert to per-motor units for the allocator
        T        = F_total / 4.0
        r        = tau_roll  / (4.0 * ARM)
        p        = tau_pitch / (4.0 * ARM)
        yaw_comp = tau_yaw   / (4.0 * THRUST2TORQUE)

        return T, r, p, yaw_comp


# ── MuJoCo helpers ─────────────────────────────────────────────────────────────

def get_state(data: mujoco.MjData) -> dict:
    """Read sensors → state dict with pos, vel, rpy [rad], gyro [rad/s]."""
    quat = data.sensor('body_quat').data.copy()    # [w, x, y, z]
    gyro = data.sensor('body_gyro').data.copy()    # rad/s body frame
    pos  = data.sensor('body_pos').data.copy()     # m world frame
    vel  = data.sensor('body_linvel').data.copy()  # m/s world frame

    w, x, y, z = quat
    roll  = math.atan2(2.0*(w*x + y*z), 1.0 - 2.0*(x*x + y*y))
    pitch = math.asin(float(np.clip(2.0*(w*y - z*x), -1.0, 1.0)))
    yaw   = math.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))

    return {'pos': pos, 'vel': vel,
            'rpy': np.array([roll, pitch, yaw]), 'gyro': gyro}


def apply_wrench(model: mujoco.MjModel, data: mujoco.MjData,
                 thrust: float, tau_roll: float,
                 tau_pitch: float, tau_yaw: float) -> None:
    """Apply body-frame wrench to the 'cf2' body via xfrc_applied (world frame)."""
    body_id = model.body('cf2').id
    xmat    = data.xmat[body_id].reshape(3, 3)
    data.xfrc_applied[body_id, :3] = xmat @ np.array([0.0, 0.0, thrust])
    data.xfrc_applied[body_id, 3:] = xmat @ np.array([tau_roll, tau_pitch, tau_yaw])


def mark_dead_motor(model: mujoco.MjModel, failed: int) -> None:
    """Turn the failed motor's propeller red in the viewer."""
    if 1 <= failed <= 4:
        geom_id = model.geom(f'prop{failed}').id
        model.geom_rgba[geom_id] = RGBA_FAIL


def reset_prop_colors(model: mujoco.MjModel) -> None:
    """Restore all propellers to healthy green."""
    for i in range(1, 5):
        model.geom_rgba[model.geom(f'prop{i}').id] = RGBA_OK


# ── per-case simulation ────────────────────────────────────────────────────────

def run_case(
    model: mujoco.MjModel,
    data:  mujoco.MjData,
    failed_motor: int,
    viewer: Optional[object],       # mujoco.viewer passive handle or None
    output_dir: Path,
    hover_z: float = HOVER_Z,
    fault_time: float = FAULT_TIME,
    verbose: bool = True,
) -> Path:
    """
    Run one complete fault-injection test case.
    Resets data, runs the full flight sequence, saves CSV.

    Returns the path to the saved CSV file.
    """
    # ── reset simulation ────────────────────────────────────────────────────
    mujoco.mj_resetData(model, data)
    reset_prop_colors(model)

    ctrl = HoverController(target_z=hover_z)
    ctrl.reset()

    dt     = model.opt.timestep          # 0.002 s
    phase  = Phase.HOVER
    ft_stab_start: Optional[float] = None
    descent_z: float = hover_z          # tracks the descending target

    rows: list[dict] = []

    # CSV fieldnames (defined once for consistent ordering)
    fieldnames = [
        't_s', 'phase',
        'm1_N', 'm2_N', 'm3_N', 'm4_N',
        'thrust_N', 'tau_roll_Nm', 'tau_pitch_Nm', 'tau_yaw_Nm',
        'roll_deg', 'pitch_deg', 'yaw_deg',
        'x_m', 'y_m', 'z_m',
        'vx_ms', 'vy_ms', 'vz_ms',
        'target_z_m', 'failed_motor',
    ]

    if verbose:
        print(f'\n{"─"*60}')
        print(f'  Case: motor {failed_motor} fails at t={FAULT_TIME:.1f}s')
        print(f'{"─"*60}')

    # ── simulation loop ─────────────────────────────────────────────────────
    while data.time < TOTAL_TIMEOUT:
        t   = data.time
        st  = get_state(data)
        z   = float(st['pos'][2])

        # ── phase transitions ─────────────────────────────────────────────
        if phase == Phase.HOVER and t >= fault_time:
            phase = Phase.FT_STAB
            ft_stab_start = t
            mark_dead_motor(model, failed_motor)
            if verbose:
                print(f'  t={t:.2f}s  FAULT INJECTED  (motor {failed_motor})')

        if phase == Phase.FT_STAB:
            elapsed_ft = t - (ft_stab_start or t)
            if elapsed_ft >= FT_STAB_TIME:
                phase = Phase.DESCENT
                descent_z = HOVER_Z
                if verbose:
                    print(f'  t={t:.2f}s  BEGIN DESCENT')

        if phase == Phase.DESCENT:
            descent_z = max(0.0, descent_z - DESCENT_RATE * dt)
            ctrl.target_z = descent_z
            if z < LAND_Z:  # noqa: SIM102
                phase = Phase.LANDED
                if verbose:
                    print(f'  t={t:.2f}s  LANDED  (z={z:.3f}m)')

        if phase == Phase.LANDED:
            # Zero all forces and stop
            data.xfrc_applied[:] = 0.0
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
            break

        # ── controller ────────────────────────────────────────────────────
        # In HOVER, ensure target is held; in FT_STAB hold hover_z
        if phase in (Phase.HOVER, Phase.FT_STAB):
            ctrl.target_z = hover_z

        T, r, p, yaw_comp = ctrl.compute(st, dt)

        # ── allocate motor forces ─────────────────────────────────────────
        if phase == Phase.HOVER:
            forces = alloc_4motor(T, r, p, yaw_comp)
            thrust, tau_roll, tau_pitch, tau_yaw = motors_to_wrench(forces)
        else:
            # FT_STAB or DESCENT — use 3-motor allocator, yaw is residual
            forces, tau_yaw = alloc_fault_tolerant(T, r, p, failed_motor)
            thrust, tau_roll, tau_pitch, _ = motors_to_wrench(forces)

        # ── apply wrench ──────────────────────────────────────────────────
        apply_wrench(model, data, thrust, tau_roll, tau_pitch, tau_yaw)

        # ── log row ───────────────────────────────────────────────────────
        rpy = st['rpy']
        vel = st['vel']
        rows.append({
            't_s':          t,
            'phase':        phase.name,
            'm1_N':         float(forces[0]),
            'm2_N':         float(forces[1]),
            'm3_N':         float(forces[2]),
            'm4_N':         float(forces[3]),
            'thrust_N':     thrust,
            'tau_roll_Nm':  tau_roll,
            'tau_pitch_Nm': tau_pitch,
            'tau_yaw_Nm':   tau_yaw,
            'roll_deg':     math.degrees(rpy[0]),
            'pitch_deg':    math.degrees(rpy[1]),
            'yaw_deg':      math.degrees(rpy[2]),
            'x_m':          float(st['pos'][0]),
            'y_m':          float(st['pos'][1]),
            'z_m':          z,
            'vx_ms':        float(vel[0]),
            'vy_ms':        float(vel[1]),
            'vz_ms':        float(vel[2]),
            'target_z_m':   ctrl.target_z,
            'failed_motor': failed_motor,
        })

        # ── step ──────────────────────────────────────────────────────────
        mujoco.mj_step(model, data)

        # ── viewer sync (every 5 steps ≈ 100 Hz visual) ───────────────────
        if viewer is not None:
            if not viewer.is_running():
                if verbose:
                    print('  Viewer closed — ending case early.')
                break
            # Update camera to track drone position
            viewer.cam.lookat[:] = st['pos']
            if int(data.time / dt) % 5 == 0:
                viewer.sync()

    # ── save CSV ────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f'motor{failed_motor}_fault.csv'
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── per-case summary ─────────────────────────────────────────────────
    if verbose and rows:
        ft_rows   = [r for r in rows if r['phase'] in ('FT_STAB', 'DESCENT')]
        hov_rows  = [r for r in rows if r['phase'] == 'HOVER']
        all_rolls  = [abs(r['roll_deg'])  for r in ft_rows] or [0.0]
        all_pitches= [abs(r['pitch_deg']) for r in ft_rows] or [0.0]
        yaw_start  = rows[0]['yaw_deg']
        yaw_end    = rows[-1]['yaw_deg']
        z_hover    = np.mean([r['z_m'] for r in hov_rows[-100:]]) if hov_rows else 0.0
        print(f'  Rows logged      : {len(rows)}  ({len(rows)*dt:.1f} s)')
        print(f'  Hover z (mean)   : {z_hover:.3f} m  (target {hover_z} m)')
        print(f'  FT max |roll|    : {max(all_rolls):.1f}°')
        print(f'  FT max |pitch|   : {max(all_pitches):.1f}°')
        print(f'  Yaw drift        : {yaw_end - yaw_start:+.1f}°  (expected in FT)')
        print(f'  Final phase      : {rows[-1]["phase"]}')
        print(f'  CSV saved        : {csv_path}')

    return csv_path


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Crazyflie MuJoCo fault-injection test — motors 1–4',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Usage')[1] if 'Usage' in __doc__ else '',
    )
    parser.add_argument(
        '--xml', type=Path, default=DEFAULT_XML,
        help=f'Path to MJCF XML (default: {DEFAULT_XML})',
    )
    parser.add_argument(
        '--cases', nargs='+', type=int, choices=[1, 2, 3, 4],
        default=[1, 2, 3, 4],
        help='Which motor failure cases to run (default: all four)',
    )
    parser.add_argument(
        '--output-dir', type=Path, default=DEFAULT_OUT,
        help=f'Directory for CSV output (default: {DEFAULT_OUT})',
    )
    parser.add_argument(
        '--headless', action='store_true',
        help='Run without MuJoCo viewer (batch mode)',
    )
    parser.add_argument(
        '--hover-z', type=float, default=HOVER_Z,
        help=f'Target hover altitude [m] (default: {HOVER_Z})',
    )
    parser.add_argument(
        '--fault-time', type=float, default=FAULT_TIME,
        help=f'Time [s] of fault injection (default: {FAULT_TIME})',
    )
    args = parser.parse_args()

    # resolve XML
    xml = args.xml.expanduser().resolve()
    if not xml.exists():
        print(f'[ERROR] XML not found: {xml}', file=sys.stderr)
        return 1

    hover_z    = args.hover_z
    fault_time = args.fault_time

    print('Crazyflie MuJoCo fault-injection test')
    print(f'  XML        : {xml}')
    print(f'  Cases      : motors {args.cases}')
    print(f'  Hover z    : {hover_z} m')
    print(f'  Fault at   : t={fault_time:.1f}s')
    print(f'  Output dir : {args.output_dir}')
    print(f'  Viewer     : {"headless" if args.headless else "enabled"}')

    # load model once — reset data between cases
    model = mujoco.MjModel.from_xml_path(str(xml))
    data  = mujoco.MjData(model)

    # set up initial prop colours (green = healthy)
    reset_prop_colors(model)

    saved: dict[int, Path] = {}

    if args.headless:
        # ── headless batch ──────────────────────────────────────────────
        for motor in args.cases:
            saved[motor] = run_case(model, data, motor,
                                    viewer=None,
                                    output_dir=args.output_dir,
                                    hover_z=hover_z,
                                    fault_time=fault_time)
    else:
        # ── single viewer, sequential cases ─────────────────────────────
        # Camera: start behind and slightly above
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance  = 2.0
            viewer.cam.elevation = -20
            viewer.cam.azimuth   = 135

            for motor in args.cases:
                if not viewer.is_running():
                    print('\nViewer closed — stopping.')
                    break

                print(f'\n[Viewer] Motor {motor} test starting — '
                      f'close viewer to skip, or wait for landing.')

                # brief pause between cases so user can see model reset
                time.sleep(0.5)
                mujoco.mj_resetData(model, data)
                viewer.sync()
                time.sleep(0.3)

                saved[motor] = run_case(model, data, motor,
                                        viewer=viewer,
                                        output_dir=args.output_dir,
                                        hover_z=hover_z,
                                        fault_time=fault_time)

                # hold the landing pose for 1 s before next case
                for _ in range(int(1.0 / model.opt.timestep)):
                    if not viewer.is_running():
                        break
                    mujoco.mj_step(model, data)
                    viewer.sync()

    # ── final summary ────────────────────────────────────────────────────
    print(f'\n{"=" * 60}')
    print('ALL CASES COMPLETE')
    print(f'{"=" * 60}')
    for motor, path in saved.items():
        print(f'  Motor {motor}: {path}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
