#!/usr/bin/env python3
"""
Crazyflie MuJoCo SITL – Fault-Tolerant 3-Motor Allocation Test
===============================================================
Implements both the standard 4-motor mixer and the closed-form
3-motor fault-tolerant allocation from power_distribution_quadrotor.c.

Usage
-----
Normal 4-motor hover:
    python3 hover_sim.py

Fault-inject motor 2 (headless, 15 s):
    python3 hover_sim.py --failed-motor 2 --duration 15 --headless

All motors:
    for m in 0 1 2 3 4; do
        python3 hover_sim.py --failed-motor $m --headless --duration 8
    done

Fault-tolerant allocation formulae (firmware variables r = roll/2, p = pitch/2)
---------------------------------------------------------------------------------
  Motor 1 fails → m2=2(T-r),  m3=2(r-p),  m4=2(T+p)
  Motor 2 fails → m1=2(T-r),  m3=2(T-p),  m4=2(r+p)
  Motor 3 fails → m1=2(p-r),  m2=2(T-p),  m4=2(T+r)
  Motor 4 fails → m1=2(T+p),  m2=-2(r+p), m3=2(T+r)
"""

import argparse
import math
import os
import sys

import mujoco
import mujoco.viewer
import numpy as np

# ── Physical constants ─────────────────────────────────────────────────────────
MASS    = 0.027          # kg  (CF2 datasheet)
G       = 9.81           # m/s²
HOVER_F = MASS * G       # N, total thrust at hover ≈ 0.2648 N

# Motor layout: X-config, arm length along diagonal = 0.046 m
# Effective moment arm = 0.046 * sin(45°) = 0.0325 m
ARM_LEN   = 0.046        # m, motor centre to CF2 centre
ARM       = ARM_LEN * math.sin(math.radians(45))  # ≈ 0.03253 m

# Thrust-to-reaction-torque ratio (from platform_defaults_cf2.h)
THRUST2TORQUE = 0.005964552  # N·m / N

# Per-motor max thrust: CF2.1 hovers at ~44% throttle → max ≈ 0.60 N total
# (0.0662 N hover per motor / 0.44 ≈ 0.150 N max per motor)
# This gives ~30% headroom for the FT allocator which needs 2×T_hover per motor.
MAX_MOTOR_F = 0.60 / 4   # N  (= 0.150 N)

# Spin directions (CCW = +1, CW = -1) for yaw reaction torque
#   M1 front-left  CCW, M2 back-left  CW, M3 back-right CCW, M4 front-right CW
SPIN = np.array([1.0, -1.0, 1.0, -1.0])

# ── Motor-to-wrench conversion ─────────────────────────────────────────────────
def motors_to_wrench(f: np.ndarray):
    """
    Convert per-motor thrust forces [m1,m2,m3,m4] (N) to body-frame wrench.
    Motor layout (looking from above, body +x forward, +y left):
        M1 front-left, M2 back-left, M3 back-right, M4 front-right
    Returns (total_thrust, tau_roll, tau_pitch, tau_yaw) in N and N·m.
    """
    thrust    = float(np.sum(f))
    tau_roll  = ARM * float((f[2] + f[3]) - (f[0] + f[1]))   # +roll = right side up
    tau_pitch = ARM * float((f[0] + f[3]) - (f[1] + f[2]))   # +pitch = front up
    tau_yaw   = THRUST2TORQUE * float(np.dot(SPIN, f))        # +yaw = CCW from above
    return thrust, tau_roll, tau_pitch, tau_yaw


# ── 4-Motor standard allocation (firmware legacy mode) ─────────────────────────
def alloc_4motor(T: float, r: float, p: float, yaw: float) -> np.ndarray:
    """
    Standard Crazyflie mixer (firmware power_distribution_quadrotor.c).
    T, r, p, yaw are in physical Newtons (per-motor level).
    r  = roll_torque  / (4 * ARM)
    p  = pitch_torque / (4 * ARM)
    yaw = yaw_torque  / (4 * THRUST2TORQUE)
    Returns per-motor forces in N, clipped to [0, MAX_MOTOR_F].
    """
    m1 = T - r + p + yaw
    m2 = T - r - p - yaw
    m3 = T + r - p + yaw
    m4 = T + r + p - yaw
    return np.clip([m1, m2, m3, m4], 0.0, MAX_MOTOR_F)


# ── 3-Motor fault-tolerant allocation ─────────────────────────────────────────
def alloc_fault_tolerant(T: float, r: float, p: float,
                         failed: int) -> tuple:
    """
    Closed-form 3-motor allocation from power_distribution_quadrotor.c.

    Prioritises exact tracking of thrust (T), roll (r) and pitch (p).
    Yaw is a residual — returned as tau_yaw_residual (N·m).

    Parameters
    ----------
    T       : mean per-motor thrust command (N)  = total_thrust / 4
    r       : roll  component (N)  = roll_torque  / (4 * ARM)
    p       : pitch component (N)  = pitch_torque / (4 * ARM)
    failed  : motor index 1-4 that has failed (0 → fall back to 4-motor)

    Returns
    -------
    forces          : np.ndarray shape (4,), per-motor thrust in N
    yaw_residual    : float, uncontrolled yaw torque in N·m
    """
    m = np.zeros(4)

    if failed == 1:
        # Motor 1 (front-left) dead. Working: 2, 3, 4
        m[1] = 2.0 * (T - r)
        m[2] = 2.0 * (r - p)
        m[3] = 2.0 * (T + p)
        yaw_res_N = -T + r - p          # per-motor level residual

    elif failed == 2:
        # Motor 2 (back-left) dead. Working: 1, 3, 4
        m[0] = 2.0 * (T - r)
        m[2] = 2.0 * (T - p)
        m[3] = 2.0 * (r + p)
        yaw_res_N = T - r - p

    elif failed == 3:
        # Motor 3 (back-right) dead. Working: 1, 2, 4
        m[0] = 2.0 * (p - r)
        m[1] = 2.0 * (T - p)
        m[3] = 2.0 * (T + r)
        yaw_res_N = p - r - T

    elif failed == 4:
        # Motor 4 (front-right) dead. Working: 1, 2, 3
        m[0] = 2.0 * (T + p)
        m[1] = -2.0 * (r + p)
        m[2] = 2.0 * (T + r)
        yaw_res_N = T + r + p

    else:
        # No fault or invalid → standard 4-motor
        return alloc_4motor(T, r, p, 0.0), 0.0

    forces = np.clip(m, 0.0, MAX_MOTOR_F)
    yaw_residual = THRUST2TORQUE * 4.0 * yaw_res_N   # convert to N·m
    return forces, yaw_residual


# ── PD Hover Controller (SI units throughout) ──────────────────────────────────
class HoverController:
    """
    Cascaded PD controller.
      Outer loop: altitude z → total thrust (N)
      Inner loop: roll / pitch angles → roll / pitch torques (N·m)
      Yaw:        yaw angle → yaw torque (N·m)
    """
    def __init__(self, target_z: float = 1.0):
        self.target_z = target_z

        # Altitude PD  (critical damping at ωn ≈ 4 rad/s)
        self.kp_z  = MASS * 16.0     # ≈ 0.432 N/m
        self.kd_z  = MASS * 8.0      # ≈ 0.216 N·s/m

        # Attitude PD  (critical damping at ωn ≈ 20 rad/s)
        # Ixx = 2.4e-5 kg·m²
        Ixx = 2.3951e-5
        Izz = 3.2347e-5
        self.kp_rp  = Ixx * 400.0   # ≈ 9.6e-3 N·m/rad
        self.kd_rp  = Ixx * 40.0    # ≈ 9.6e-4 N·m·s/rad
        self.kp_yaw = Izz * 25.0    # ≈ 8.1e-4 N·m/rad
        self.kd_yaw = Izz * 10.0    # ≈ 3.2e-4 N·m·s/rad

    def compute(self, state: dict) -> tuple:
        """
        Returns (T, r, p, yaw_comp) — all in Newtons (per-motor scale).
        T        = total_thrust / 4
        r        = roll_torque  / (4 * ARM)
        p        = pitch_torque / (4 * ARM)
        yaw_comp = yaw_torque   / (4 * THRUST2TORQUE)
        """
        z    = state['pos'][2]
        vz   = state['vel'][2]
        roll, pitch, yaw = state['rpy']
        ox, oy, oz = state['gyro']

        # Altitude → total thrust
        F_total = (HOVER_F
                   + self.kp_z  * (self.target_z - z)
                   - self.kd_z  * vz)
        F_total = float(np.clip(F_total, 0.0, 0.35))

        # Attitude → torques
        tau_roll  = (-self.kp_rp  * roll  - self.kd_rp  * ox)
        tau_pitch = (-self.kp_rp  * pitch - self.kd_rp  * oy)
        tau_yaw   = (-self.kp_yaw * yaw   - self.kd_yaw * oz)

        # Convert to per-motor units
        T        = F_total / 4.0
        r        = tau_roll  / (4.0 * ARM)
        p        = tau_pitch / (4.0 * ARM)
        yaw_comp = tau_yaw   / (4.0 * THRUST2TORQUE)

        return T, r, p, yaw_comp


# ── State extraction ────────────────────────────────────────────────────────────
def get_state(data) -> dict:
    """Read sensors and return a state dict."""
    quat = data.sensor('body_quat').data.copy()    # [w, x, y, z]
    gyro = data.sensor('body_gyro').data.copy()    # rad/s, body frame
    pos  = data.sensor('body_pos').data.copy()     # m, world frame
    vel  = data.sensor('body_linvel').data.copy()  # m/s, world frame

    # Quaternion → Euler (ZYX convention)
    w, x, y, z = quat
    roll  = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = math.asin(float(np.clip(2*(w*y - z*x), -1.0, 1.0)))
    yaw   = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    return {
        'pos': pos,
        'vel': vel,
        'rpy': np.array([roll, pitch, yaw]),
        'gyro': gyro,
    }


# ── Apply body-frame wrench via xfrc_applied ───────────────────────────────────
def apply_wrench(model, data, body_name: str,
                 thrust: float, tau_roll: float,
                 tau_pitch: float, tau_yaw: float):
    """
    Apply a body-frame wrench (thrust in body +z, torques in body frame)
    to a MuJoCo body using xfrc_applied (world frame).
    """
    body_id = model.body(body_name).id
    # Body-to-world rotation matrix
    xmat = data.xmat[body_id].reshape(3, 3)

    # Rotate thrust (body +z) and torques to world frame
    thrust_world = xmat @ np.array([0.0, 0.0, thrust])
    tau_world    = xmat @ np.array([tau_roll, tau_pitch, tau_yaw])

    data.xfrc_applied[body_id, :3] = thrust_world
    data.xfrc_applied[body_id, 3:] = tau_world


# ── Main simulation ─────────────────────────────────────────────────────────────
def run_sim(failed_motor: int = 0,
            fault_inject_time: float = 3.0,
            duration: float = 10.0,
            headless: bool = False,
            xml_path: str = None) -> dict:
    """
    Run the hover simulation.

    Parameters
    ----------
    failed_motor      : 0 = all healthy; 1-4 = which motor fails at fault_inject_time
    fault_inject_time : seconds into the simulation before the fault is injected
    duration          : total simulation duration in seconds
    headless          : if True, no viewer window
    xml_path          : path to MJCF file (defaults to cf2_sim.xml next to this script)

    Returns
    -------
    log dict with keys: t, z, roll, pitch, yaw, thrust, active_fault
    """
    if xml_path is None:
        xml_path = os.path.join(os.path.dirname(__file__), 'cf2_sim.xml')

    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    ctrl = HoverController(target_z=1.0)
    dt   = model.opt.timestep           # 0.002 s
    steps = int(duration / dt)

    log = {
        't': [], 'z': [], 'roll': [], 'pitch': [], 'yaw': [],
        'thrust': [], 'active_fault': [],
        'tau_roll': [], 'tau_pitch': [], 'tau_yaw_actual': [],
    }

    def step_fn(model, data):
        t_now = data.time
        current_fault = failed_motor if t_now >= fault_inject_time else 0

        state  = get_state(data)
        T, r, p, yaw_comp = ctrl.compute(state)

        if current_fault == 0:
            forces = alloc_4motor(T, r, p, yaw_comp)
            thrust, tau_roll, tau_pitch, tau_yaw = motors_to_wrench(forces)
        else:
            forces, tau_yaw_residual = alloc_fault_tolerant(T, r, p, current_fault)
            thrust, tau_roll, tau_pitch, _ = motors_to_wrench(forces)
            tau_yaw = tau_yaw_residual  # yaw is now uncontrolled residual

        apply_wrench(model, data, 'cf2', thrust, tau_roll, tau_pitch, tau_yaw)

        # Log
        rpy = state['rpy']
        log['t'].append(t_now)
        log['z'].append(state['pos'][2])
        log['roll'].append(math.degrees(rpy[0]))
        log['pitch'].append(math.degrees(rpy[1]))
        log['yaw'].append(math.degrees(rpy[2]))
        log['thrust'].append(thrust)
        log['active_fault'].append(current_fault)
        log['tau_roll'].append(tau_roll)
        log['tau_pitch'].append(tau_pitch)
        log['tau_yaw_actual'].append(tau_yaw)

    if headless:
        for _ in range(steps):
            step_fn(model, data)
            mujoco.mj_step(model, data)
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = [0, 0, 0.8]
            viewer.cam.distance  = 2.5
            viewer.cam.elevation = -20
            for i in range(steps):
                step_fn(model, data)
                mujoco.mj_step(model, data)
                if i % 10 == 0:
                    viewer.sync()
                if not viewer.is_running():
                    break

    return log


# ── CLI entry-point ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Crazyflie MuJoCo SITL with fault-tolerant 3-motor allocation')
    parser.add_argument('--failed-motor', type=int, default=0,
                        choices=[0, 1, 2, 3, 4],
                        help='Motor to fail (0 = healthy, 1-4 = motor index)')
    parser.add_argument('--fault-time', type=float, default=3.0,
                        help='Time (s) at which fault is injected (default: 3.0)')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Total simulation duration in seconds (default: 10)')
    parser.add_argument('--headless', action='store_true',
                        help='Run without GUI viewer')
    parser.add_argument('--xml', type=str, default=None,
                        help='Path to MJCF XML (default: cf2_sim.xml next to script)')
    args = parser.parse_args()

    if args.failed_motor == 0:
        mode_str = '4-motor (all healthy)'
    else:
        mode_str = (f'3-motor FT — motor {args.failed_motor} fails '
                    f'at t={args.fault_time:.1f}s')

    print(f'Mode    : {mode_str}')
    print(f'Duration: {args.duration:.1f} s')
    print(f'Viewer  : {"headless" if args.headless else "interactive"}')
    print()

    log = run_sim(
        failed_motor=args.failed_motor,
        fault_inject_time=args.fault_time,
        duration=args.duration,
        headless=args.headless,
        xml_path=args.xml,
    )

    # ── Results summary ────────────────────────────────────────────────────────
    t      = np.array(log['t'])
    z      = np.array(log['z'])
    roll   = np.array(log['roll'])
    pitch  = np.array(log['pitch'])
    yaw    = np.array(log['yaw'])

    # Evaluate only the post-fault steady-state window (last 4 s)
    ss_mask = t >= (t[-1] - 4.0)
    z_ss    = z[ss_mask]
    r_ss    = roll[ss_mask]
    p_ss    = pitch[ss_mask]

    print('─' * 52)
    print(f'Results  [{mode_str}]')
    print('─' * 52)
    print(f'  Final altitude      : {z[-1]:.3f} m   (target 1.000 m)')
    print(f'  Altitude RMSE (ss)  : {np.sqrt(np.mean((z_ss - 1.0)**2)):.4f} m')
    print(f'  Max |roll|  (ss)    : {np.max(np.abs(r_ss)):.2f}°')
    print(f'  Max |pitch| (ss)    : {np.max(np.abs(p_ss)):.2f}°')
    print(f'  Yaw drift           : {yaw[-1] - yaw[0]:.1f}°')
    stable = (np.max(np.abs(r_ss)) < 15.0 and np.max(np.abs(p_ss)) < 15.0
              and z[-1] > 0.3)
    print(f'  Stable              : {stable}')

    if args.failed_motor != 0:
        print()
        print('  Note: yaw is uncontrolled in 3-motor FT mode (expected drift).')
        print('  Thrust, roll, and pitch are maintained by the closed-form allocator.')

    return 0 if stable else 1


if __name__ == '__main__':
    sys.exit(main())
