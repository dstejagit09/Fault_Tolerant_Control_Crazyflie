#!/usr/bin/env python3
"""
simu_demo.py — Crazyflie Fault-Tolerant Flight Demo for Presentation
=====================================================================
Opens the MuJoCo "Drone scene" viewer and flies through the full
fault-injection-and-recovery sequence at real-time speed.

A 4-second countdown is printed after the viewer opens so you have time to:
  • Hide terminal panels  (Tab / Shift+Tab in the MuJoCo viewer)
  • Fullscreen the viewer (F11 or drag to fill screen)
  • Start your screen recorder

Flight sequence (per case)
--------------------------
  0 s              TAKEOFF  : linear ramp 0 → 0.5 m over 2 s
  2 s              HOVER    : stable 4-motor hover for 3 s
  5 s              FAULT    : motor N killed → prop turns red
  5 s – 7 s        FT_HOLD  : 3-motor stabilisation at 0.5 m
  7 s – ~12 s      DESCENT  : controlled descent at 0.10 m/s
  ~12 s            LANDED   : stop setpoint

All phase transitions are printed to the terminal in real time.
The camera smoothly tracks the drone and stays at:
    distance = 2.0 m, elevation = -20°, azimuth = 135°

Usage
-----
# Single case (default motor 1):
python3 simu_demo.py

# Specific motor:
python3 simu_demo.py --motor 2

# All 4 cases sequentially:
python3 simu_demo.py --all

# Custom fault motor with no pause between --all cases:
python3 simu_demo.py --all --pause 0
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from enum import Enum, auto
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
XML_PATH = _HERE / 'cf2_sim.xml'

# ── physical constants (CF2, match hover_sim.py) ───────────────────────────────
MASS           = 0.027
G              = 9.81
HOVER_F        = MASS * G
ARM_LEN        = 0.046
ARM            = ARM_LEN * math.sin(math.radians(45))
THRUST2TORQUE  = 0.005964552
MAX_MOTOR_F    = 0.60 / 4
SPIN           = np.array([1.0, -1.0, 1.0, -1.0])
Ixx            = 2.3951e-5
Izz            = 3.2347e-5

# ── flight timing ──────────────────────────────────────────────────────────────
TAKEOFF_RAMP_S  = 3.0    # s — ramp from 0 to HOVER_Z
PRE_FAULT_S     = 3.0    # s — stable 4-motor hover before fault
FT_HOLD_S       = 2.0    # s — 3-motor hold after fault before descent
DESCENT_RATE    = 0.10   # m/s — controlled descent speed
HOVER_Z         = 1.0    # m
LAND_Z          = 0.06   # m — below this → landed
GROUND_Z        = 0.01   # m — drone spawn height (3 mm above floor, no penetration)
INTER_CASE_S    = 2.0    # s — pause between --all cases
STARTUP_DELAY_S = 4      # s — countdown after viewer opens, before flight begins

# ── camera ─────────────────────────────────────────────────────────────────────
CAM_DISTANCE    = 2.0
CAM_ELEVATION   = -20.0
CAM_AZIMUTH     = 135.0
CAM_LERP        = 0.08   # smoothing factor for lookat tracking

# ── prop colours ───────────────────────────────────────────────────────────────
RGBA_OK   = np.array([0.20, 0.80, 0.20, 1.0])
RGBA_FAIL = np.array([0.90, 0.10, 0.10, 1.0])


# ══════════════════════════════════════════════════════════════════════════════
# Flight phases
# ══════════════════════════════════════════════════════════════════════════════

class Phase(Enum):
    TAKEOFF  = auto()
    HOVER    = auto()
    FAULT    = auto()   # prop turns red, FT allocator kicks in
    FT_HOLD  = auto()   # 3-motor stabilisation hold
    DESCENT  = auto()
    LANDED   = auto()


# ══════════════════════════════════════════════════════════════════════════════
# Allocators
# ══════════════════════════════════════════════════════════════════════════════

def alloc_4motor(T, r, p, yaw):
    return np.clip([T-r+p+yaw, T-r-p-yaw, T+r-p+yaw, T+r+p-yaw], 0.0, MAX_MOTOR_F)


def alloc_ft(T, r, p, failed):
    m = np.zeros(4)
    if   failed == 1: m[1]=2*(T-r); m[2]=2*(r-p); m[3]=2*(T+p); yN=-T+r-p
    elif failed == 2: m[0]=2*(T-r); m[2]=2*(T-p); m[3]=2*(r+p); yN=T-r-p
    elif failed == 3: m[0]=2*(p-r); m[1]=2*(T-p); m[3]=2*(T+r); yN=p-r-T
    elif failed == 4: m[0]=2*(T+p); m[1]=-2*(r+p);m[2]=2*(T+r); yN=T+r+p
    else:             return alloc_4motor(T,r,p,0), 0.0
    return np.clip(m, 0.0, MAX_MOTOR_F), THRUST2TORQUE * 4 * yN


def motors_to_wrench(f):
    thrust    = float(f.sum())
    tau_roll  = ARM * float((f[2]+f[3]) - (f[0]+f[1]))
    tau_pitch = ARM * float((f[0]+f[3]) - (f[1]+f[2]))
    tau_yaw   = THRUST2TORQUE * float(SPIN @ f)
    return thrust, tau_roll, tau_pitch, tau_yaw


# ══════════════════════════════════════════════════════════════════════════════
# Controller
# ══════════════════════════════════════════════════════════════════════════════

class HoverController:
    def __init__(self, target_z=HOVER_Z):
        self.target_z = target_z
        self.kp_z  = MASS * 25.0
        self.ki_z  = MASS * 4.0
        self.kd_z  = MASS * 10.0
        self.kp_rp = Ixx * 400.0
        self.kd_rp = Ixx * 40.0
        self.kp_yaw= Izz * 25.0
        self.kd_yaw= Izz * 10.0
        self._zi   = 0.0

    def reset(self):
        self._zi = 0.0

    def compute(self, st, dt):
        z   = float(st['pos'][2])
        vz  = float(st['vel'][2])
        roll, pitch, yaw = st['rpy']
        ox, oy, oz = st['gyro']
        ze         = self.target_z - z
        self._zi   = float(np.clip(self._zi + ze * dt, -0.5, 0.5))
        Ft = float(np.clip(
            HOVER_F + self.kp_z*ze + self.ki_z*self._zi - self.kd_z*vz,
            0.0, MAX_MOTOR_F*4))
        tr = -self.kp_rp*roll  - self.kd_rp*ox
        tp = -self.kp_rp*pitch - self.kd_rp*oy
        ty = -self.kp_yaw*yaw  - self.kd_yaw*oz
        return Ft/4, tr/(4*ARM), tp/(4*ARM), ty/(4*THRUST2TORQUE)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_state(data):
    quat = data.sensor('body_quat').data.copy()
    gyro = data.sensor('body_gyro').data.copy()
    pos  = data.sensor('body_pos').data.copy()
    vel  = data.sensor('body_linvel').data.copy()
    w,x,y,z = quat
    roll  = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = math.asin(float(np.clip(2*(w*y-z*x),-1,1)))
    yaw   = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return {'pos':pos,'vel':vel,'rpy':np.array([roll,pitch,yaw]),'gyro':gyro}


def apply_wrench(model, data, thrust, tau_roll, tau_pitch, tau_yaw):
    bid  = model.body('cf2').id
    xmat = data.xmat[bid].reshape(3,3)
    data.xfrc_applied[bid,:3] = xmat @ np.array([0,0,thrust])
    data.xfrc_applied[bid,3:] = xmat @ np.array([tau_roll,tau_pitch,tau_yaw])


def reset_prop_colors(model):
    for i in range(1, 5):
        model.geom_rgba[model.geom(f'prop{i}').id] = RGBA_OK.copy()


def mark_failed_prop(model, motor):
    model.geom_rgba[model.geom(f'prop{motor}').id] = RGBA_FAIL.copy()


def _phase_banner(phase_name, motor=None, extra=''):
    pad = 60
    if motor is not None:
        line = f'  [ {phase_name} ]  motor {motor} failed  {extra}'
    else:
        line = f'  [ {phase_name} ]  {extra}'
    print(f'\n{"─"*pad}')
    print(line)
    print(f'{"─"*pad}')


def _startup_countdown(viewer, seconds: int = STARTUP_DELAY_S) -> bool:
    """
    Print a countdown to terminal while keeping the viewer alive.
    Returns False if the viewer was closed during the countdown.
    """
    print()
    print('  ┌─────────────────────────────────────────────────┐')
    print('  │  MuJoCo viewer is open.  You have time to:      │')
    print('  │    • Fullscreen / resize the viewer window       │')
    print('  │    • Hide terminal panels  (Tab in viewer)       │')
    print('  │    • Start your screen recorder                  │')
    print('  └─────────────────────────────────────────────────┘')
    for n in range(seconds, 0, -1):
        if not viewer.is_running():
            return False
        print(f'  Starting in  {n}...', end='\r', flush=True)
        time.sleep(1.0)
        viewer.sync()
    print(f'  Starting in  0...  GO!          ')
    print()
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Single-case simulation (runs inside an already-open viewer)
# ══════════════════════════════════════════════════════════════════════════════

def run_case(model, data, viewer, failed_motor: int) -> None:
    """
    Execute one complete fault-injection demo case.

    The simulation runs at real-time speed by sleeping to align wall-clock
    with sim time.  The viewer camera smoothly tracks the drone position.

    Parameters
    ----------
    model        : loaded MjModel (not reset here — caller resets between cases)
    data         : MjData
    viewer       : passive viewer handle
    failed_motor : 1-4
    """
    mujoco.mj_resetData(model, data)
    # Place drone on the ground instead of the XML default spawn height.
    # qpos layout for freejoint: [x, y, z, qw, qx, qy, qz]
    # Body PCB extends 0.007 m below centre; 0.01 m gives 3 mm floor clearance.
    data.qpos[2] = GROUND_Z
    mujoco.mj_forward(model, data)   # propagate position to xpos/xmat/etc.
    reset_prop_colors(model)
    viewer.sync()

    ctrl          = HoverController(target_z=HOVER_Z)
    ctrl.reset()
    dt            = float(model.opt.timestep)   # 0.002 s

    phase         = Phase.TAKEOFF
    fault_t       = TAKEOFF_RAMP_S + PRE_FAULT_S
    ft_hold_end   = fault_t + FT_HOLD_S
    descent_z     = HOVER_Z

    # Initialise camera lookat at ground level so the viewer sees
    # the drone sitting on the floor before takeoff begins.
    viewer.cam.lookat[0] = 0.0
    viewer.cam.lookat[1] = 0.0
    viewer.cam.lookat[2] = 0.0
    viewer.cam.distance  = CAM_DISTANCE
    viewer.cam.elevation = CAM_ELEVATION
    viewer.cam.azimuth   = CAM_AZIMUTH

    _phase_banner('TAKEOFF', extra=f'0 → {HOVER_Z} m over {TAKEOFF_RAMP_S:.0f} s')

    wall_start  = time.perf_counter()
    step_count  = 0
    last_phase  = phase
    SYNC_EVERY  = 5   # sync viewer every N physics steps (10 ms @ 2 ms/step)

    while viewer.is_running():
        t   = data.time
        st  = get_state(data)
        z   = float(st['pos'][2])

        # ── phase transitions ──────────────────────────────────────────────
        if phase == Phase.TAKEOFF and t >= TAKEOFF_RAMP_S:
            phase = Phase.HOVER
            ctrl.target_z = HOVER_Z
            ctrl.reset()

        if phase == Phase.HOVER and t >= fault_t:
            phase = Phase.FAULT

        if phase == Phase.FAULT:
            mark_failed_prop(model, failed_motor)
            phase = Phase.FT_HOLD

        if phase == Phase.FT_HOLD and t >= ft_hold_end:
            phase = Phase.DESCENT
            descent_z = HOVER_Z

        if phase == Phase.DESCENT:
            descent_z  = max(0.0, descent_z - DESCENT_RATE * dt)
            ctrl.target_z = descent_z
            if z < LAND_Z:
                phase = Phase.LANDED

        if phase == Phase.LANDED:
            data.xfrc_applied[:] = 0.0
            mujoco.mj_step(model, data)
            viewer.cam.lookat[:] += CAM_LERP * (st['pos'] - viewer.cam.lookat[:])
            viewer.sync()
            break

        # Print phase banner on transition (only once per phase)
        if phase != last_phase:
            last_phase = phase
            if phase == Phase.HOVER:
                _phase_banner('HOVER', extra=f'{HOVER_Z} m  ({PRE_FAULT_S:.0f} s until fault)')
            elif phase == Phase.FT_HOLD:
                _phase_banner('FAULT INJECTED', motor=failed_motor,
                              extra=f't={t:.2f}s  → 3-motor FT allocator')
            elif phase == Phase.DESCENT:
                _phase_banner('DESCENT', extra=f'at {DESCENT_RATE} m/s')

        # ── set altitude target during takeoff ramp ────────────────────────
        if phase == Phase.TAKEOFF:
            ctrl.target_z = HOVER_Z * min(1.0, t / TAKEOFF_RAMP_S)

        # ── controller + allocator ─────────────────────────────────────────
        T, r, p, yc = ctrl.compute(st, dt)

        if phase in (Phase.TAKEOFF, Phase.HOVER):
            forces = alloc_4motor(T, r, p, yc)
            thrust, tr, tp, ty = motors_to_wrench(forces)
        else:
            forces, ty = alloc_ft(T, r, p, failed_motor)
            thrust, tr, tp, _  = motors_to_wrench(forces)

        apply_wrench(model, data, thrust, tr, tp, ty)

        # ── step physics ───────────────────────────────────────────────────
        mujoco.mj_step(model, data)
        step_count += 1

        # ── real-time pacing ───────────────────────────────────────────────
        if step_count % SYNC_EVERY == 0:
            # Smooth camera: lerp lookat towards drone position
            viewer.cam.lookat[:] += CAM_LERP * (st['pos'] - viewer.cam.lookat[:])
            viewer.sync()

            # Sleep to match wall clock to sim time
            wall_elapsed = time.perf_counter() - wall_start
            sim_elapsed  = data.time
            slack        = sim_elapsed - wall_elapsed
            if slack > 0.001:
                time.sleep(slack * 0.95)   # ×0.95 to avoid over-sleeping

    if phase == Phase.LANDED:
        _phase_banner('LANDED', motor=failed_motor,
                      extra=f'final z={z:.3f} m  t={data.time:.2f}s')
    else:
        print('\n[demo] Viewer closed — case ended early.')


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Crazyflie FT flight demo for presentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--motor', type=int, default=1, choices=[1,2,3,4],
        help='Motor to fail (1-4)',
    )
    parser.add_argument(
        '--all', dest='run_all', action='store_true',
        help='Run all 4 motor-failure cases sequentially',
    )
    parser.add_argument(
        '--pause', type=float, default=INTER_CASE_S,
        help='Pause (s) between cases when using --all',
    )
    parser.add_argument(
        '--xml', type=Path, default=XML_PATH,
        help='MJCF XML path',
    )
    args = parser.parse_args()

    xml = args.xml.expanduser().resolve()
    if not xml.exists():
        print(f'[ERROR] XML not found: {xml}', file=sys.stderr)
        return 1

    cases = [1, 2, 3, 4] if args.run_all else [args.motor]

    print('=' * 60)
    print('  Crazyflie Fault-Tolerant Demo')
    print(f'  Cases: motors {cases}')
    print(f'  XML  : {xml}')
    print()
    print('  Flight sequence per case:')
    print(f'    0 s → {TAKEOFF_RAMP_S:.0f} s   TAKEOFF  (0 → {HOVER_Z} m)')
    print(f'    {TAKEOFF_RAMP_S:.0f} s → {TAKEOFF_RAMP_S+PRE_FAULT_S:.0f} s   HOVER    (4-motor)')
    print(f'    {TAKEOFF_RAMP_S+PRE_FAULT_S:.0f} s          FAULT    (prop turns red)')
    print(f'    +{FT_HOLD_S:.0f} s          FT_HOLD  (3-motor stabilise)')
    print(f'    +descent   DESCENT  ({DESCENT_RATE} m/s → land)')
    print()
    print('  Camera: distance=2.0  elevation=-20°  azimuth=135°')
    print('  Speed : real-time')
    print('=' * 60)

    model = mujoco.MjModel.from_xml_path(str(xml))
    data  = mujoco.MjData(model)
    reset_prop_colors(model)

    with mujoco.viewer.launch_passive(
            model, data,
            show_left_ui=False,
            show_right_ui=False) as viewer:
        viewer.cam.distance  = CAM_DISTANCE
        viewer.cam.elevation = CAM_ELEVATION
        viewer.cam.azimuth   = CAM_AZIMUTH
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.0])

        # ── 4-second startup countdown ─────────────────────────────────────
        # Gives the user time to fullscreen, hide panels, start recorder.
        if not _startup_countdown(viewer, STARTUP_DELAY_S):
            print('\n[demo] Viewer closed during countdown — exiting.')
            return 0

        for idx, motor in enumerate(cases):
            if not viewer.is_running():
                break

            print(f'\n{"="*60}')
            print(f'  CASE {idx+1}/{len(cases)}  — motor {motor} failure')
            print(f'{"="*60}')

            run_case(model, data, viewer, motor)

            if idx < len(cases) - 1 and args.pause > 0:
                print(f'\n  Pausing {args.pause:.0f} s before next case …')
                t0 = time.perf_counter()
                while time.perf_counter() - t0 < args.pause:
                    if not viewer.is_running():
                        break
                    time.sleep(0.05)
                    viewer.sync()

    print('\n[demo] Complete.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
