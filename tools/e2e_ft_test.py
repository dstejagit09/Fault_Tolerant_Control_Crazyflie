#!/usr/bin/env python3
"""
e2e_ft_test.py — End-to-end Gamma FDI + fault-tolerant landing test
====================================================================
Combines the Gamma FDI network, CrazySim SITL, and MuJoCo simulation into
one end-to-end validation of the full detection-and-recovery pipeline:

    1. Take off and hover at 0.5 m
    2. At t=FAULT_TIME, externally kill motor 2
       • SITL:   set powerDist.failedMotor=2 (disables the motor in the sim)
       • MuJoCo: zero motor-2 force directly in physics — runs 4-motor alloc
                 during the blind window so we see the instability
    3. Gamma FDI monitors the IMU stream and detects the fault
       • SITL:   mock model (tunable detection lag)
       • MuJoCo: sim-time mock (parameterisable detection lag, rising confidence
                 curve from t_fault to t_detect)
    4. Once FDI declares a fault (confidence ≥ threshold):
       • SITL:   FDI records detection time (firmware already uses FT via param)
       • MuJoCo: FDI triggers the switch from 4-motor to 3-motor FT allocator
    5. Vehicle descends and lands on 3 motors
    6. All data logged to CSV, timeline plot saved to results/

Output
------
  results/e2e_sitl_motor2.csv   — SITL detailed log
  results/e2e_mujoco_motor2.csv — MuJoCo detailed log
  results/e2e_timeline_sitl.png  — 5-panel SITL timeline
  results/e2e_timeline_mujoco.png— 5-panel MuJoCo timeline
  results/e2e_comparison.png     — Side-by-side SITL vs MuJoCo

Usage
-----
# Both backends (default):
python3 e2e_ft_test.py

# SITL only (requires CrazySim running):
python3 e2e_ft_test.py --backend sitl --uri udp://0.0.0.0:19950

# MuJoCo only, headless, custom output:
python3 e2e_ft_test.py --backend mujoco --headless --output-dir /tmp/e2e_logs

# Custom fault motor, detection lag, confidence threshold:
python3 e2e_ft_test.py --backend mujoco --fault-motor 3 --detect-lag 0.6 --threshold 0.85
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import numpy as np

# ── local imports ──────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from gamma_fdi import GammaFDI

# ── output paths ───────────────────────────────────────────────────────────────
_REPO_ROOT   = _HERE.parent
RESULTS_DIR  = _REPO_ROOT / 'results'
MJ_XML_PATH  = _HERE / 'mujoco_sim' / 'cf2_sim.xml'

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# Shared constants
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_URI       = 'udp://0.0.0.0:19950'
HOVER_HEIGHT_M    = 0.5       # m
FAULT_TIME_S      = 5.0       # s after takeoff begins
STAB_WAIT_S       = 3.0       # s of stabilisation hover before fault
DETECT_LAG_S      = 0.40      # s — simulated FDI detection latency after fault
FT_HOLD_S         = 1.5       # s of FT hover before descent begins
DESCENT_RATE_MS   = 0.10      # m/s
LAND_THRESHOLD_M  = 0.06      # m — consider landed below this
SAFETY_ANGLE_DEG  = 55.0      # roll/pitch kill threshold
LOG_PERIOD_MS     = 20        # logging period (50 Hz)
CONFIDENCE_THRESH = 0.80      # FDI confidence threshold

DEG2RAD  = math.pi / 180.0
G_MS2    = 9.80665

# CSV fieldnames shared by both backends (columns absent in a backend are empty)
E2E_FIELDNAMES = [
    'backend',          # 'sitl' or 'mujoco'
    't_s',              # elapsed seconds from connection/sim start
    'phase',
    # FDI
    'fdi_label',        # 0-4
    'fdi_confidence',   # [0, 1]
    'fdi_detected',     # bool — True once threshold crossed
    't_fault_s',        # time of fault injection
    't_detect_s',       # time of FDI detection
    # IMU fed to FDI (SI units)
    'gyro_x', 'gyro_y', 'gyro_z',
    'acc_x',  'acc_y',  'acc_z',
    # Motors (normalised 0-1 for both backends)
    'm1_norm', 'm2_norm', 'm3_norm', 'm4_norm',
    # Attitude
    'roll_deg', 'pitch_deg', 'yaw_deg',
    # Altitude
    'z_m',
    # FT allocator (firmware / computed)
    'ft_active',
    'ft_failed_motor',
    'ft_residual_yaw',
]

# ── Phase labels ───────────────────────────────────────────────────────────────
class Phase(Enum):
    TAKEOFF      = 'takeoff'
    HOVER        = 'hover'
    FAULT_BLIND  = 'fault_blind'   # fault injected, FDI not yet confirmed
    FT_RECOVERY  = 'ft_recovery'   # FDI confirmed, FT allocator active
    DESCENT      = 'descent'
    LANDED       = 'landed'


# ══════════════════════════════════════════════════════════════════════════════
# SITL backend (cflib / CrazySim)
# ══════════════════════════════════════════════════════════════════════════════

# ── cflib log variable tables ──────────────────────────────────────────────────
# Split into two configs to stay within the 26-byte cflib log-packet limit.

# Config A: IMU data for FDI  (6 floats = 24 bytes)
IMU_LOG_VARS = [
    ('gyro.x', 'float'),   # °/s
    ('gyro.y', 'float'),
    ('gyro.z', 'float'),
    ('acc.x',  'float'),   # g
    ('acc.y',  'float'),
    ('acc.z',  'float'),
]

# Config B: State for logging  (4×uint16 + 2×uint8 + 4×float = 26 bytes exactly)
STATE_LOG_VARS = [
    ('motor.m1',          'uint16_t'),
    ('motor.m2',          'uint16_t'),
    ('motor.m3',          'uint16_t'),
    ('motor.m4',          'uint16_t'),
    ('ftAlloc.active',    'uint8_t'),
    ('ftAlloc.failedMotor','uint8_t'),
    ('stabilizer.roll',   'float'),
    ('stabilizer.pitch',  'float'),
    ('stabilizer.yaw',    'float'),
    ('stateEstimate.z',   'float'),
]
PWM_MAX = 65535.0


class SITLBackend:
    """
    CrazySim SITL end-to-end test.

    Flight sequence:
        takeoff → stabilise → [fault inject @ FAULT_TIME_S] →
        FDI monitoring (mock, blind window) → FDI confirms →
        FT recovery hold → descent → land
    """

    def __init__(
        self,
        uri:          str,
        fault_motor:  int   = 2,
        detect_lag_s: float = DETECT_LAG_S,
        threshold:    float = CONFIDENCE_THRESH,
        output_dir:   Path  = RESULTS_DIR,
    ):
        self.uri          = uri
        self.fault_motor  = fault_motor
        self.detect_lag_s = detect_lag_s
        self.threshold    = threshold
        self.output_dir   = Path(output_dir)

        self._rows: list[dict]   = []
        self._lock               = threading.Lock()
        self._phase              = Phase.TAKEOFF
        self._t_start            = 0.0
        self._t_fault_s: float   = 0.0
        self._t_detect_s: float  = 0.0
        self._fdi_detected       = False
        self._safety_killed      = False

        # shared IMU state (written from IMU callback, read from state callback)
        self._imu_sample         = [0.0] * 6
        self._fdi_label          = 0
        self._fdi_conf           = 0.0

    # ── public entry point ────────────────────────────────────────────────────

    def run(self) -> Optional[Path]:
        """Run the full test. Returns CSV path or None on failure."""
        try:
            import cflib.crtp
            from cflib.crazyflie import Crazyflie
            from cflib.crazyflie.log import LogConfig
            from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
            from cflib.crazyflie.syncLogger import SyncLogger
            from cflib.utils import uri_helper
        except ImportError:
            print('[SITL] cflib not installed — pip install cflib')
            return None

        fdi = GammaFDI(
            model_path=None,   # mock
            confidence_threshold=self.threshold,
            mock_fault_delay_s=STAB_WAIT_S + self.detect_lag_s,
            mock_fault_motor=self.fault_motor,
        )
        fdi.load_model()

        cflib.crtp.init_drivers(enable_debug_driver=False)
        print(f'[SITL] Connecting to {self.uri} …')

        try:
            with SyncCrazyflie(self.uri, cf=Crazyflie(rw_cache='./cf_cache')) as scf:
                cf = scf.cf

                # reset any stale fault
                cf.param.set_value('powerDist.failedMotor', '0')
                time.sleep(0.3)

                # wait for Kalman filter
                self._wait_for_estimator(scf, SyncLogger, LogConfig)
                self._t_start = time.monotonic()

                # build log configs
                lc_imu   = self._build_log(cf, IMU_LOG_VARS,   'E2E_IMU',   LOG_PERIOD_MS)
                lc_state = self._build_log(cf, STATE_LOG_VARS, 'E2E_State', LOG_PERIOD_MS)

                # safety monitor (updated from state callback)
                safety = _SafetyMonitor(cf)

                def imu_cb(ts, data, _lc):
                    """Convert units, run FDI, update shared state."""
                    gx = data.get('gyro.x', 0.0) * DEG2RAD
                    gy = data.get('gyro.y', 0.0) * DEG2RAD
                    gz = data.get('gyro.z', 0.0) * DEG2RAD
                    ax = data.get('acc.x',  0.0) * G_MS2
                    ay = data.get('acc.y',  0.0) * G_MS2
                    az = data.get('acc.z',  0.0) * G_MS2
                    sample = [gx, gy, gz, ax, ay, az]
                    label, conf = fdi.predict(sample)
                    with self._lock:
                        self._imu_sample = sample
                        self._fdi_label  = label
                        self._fdi_conf   = conf
                        if (not self._fdi_detected
                                and label != 0
                                and conf >= self.threshold):
                            self._fdi_detected = True
                            self._t_detect_s   = time.monotonic() - self._t_start
                            print(
                                f'[SITL] *** FDI detected motor {label} '
                                f'conf={conf:.3f}  '
                                f'latency={(self._t_detect_s - self._t_fault_s)*1000:.0f} ms'
                            )

                def state_cb(ts, data, _lc):
                    """Log state row."""
                    roll  = data.get('stabilizer.roll',  0.0)
                    pitch = data.get('stabilizer.pitch', 0.0)
                    safety.update_attitude(roll, pitch)

                    t = time.monotonic() - self._t_start
                    with self._lock:
                        imu    = list(self._imu_sample)
                        label  = self._fdi_label
                        conf   = self._fdi_conf
                        det    = self._fdi_detected
                        t_f    = self._t_fault_s
                        t_d    = self._t_detect_s
                        phase  = self._phase.value

                    m1 = data.get('motor.m1', 0) / PWM_MAX
                    m2 = data.get('motor.m2', 0) / PWM_MAX
                    m3 = data.get('motor.m3', 0) / PWM_MAX
                    m4 = data.get('motor.m4', 0) / PWM_MAX
                    self._rows.append({
                        'backend':        'sitl',
                        't_s':            f'{t:.4f}',
                        'phase':          phase,
                        'fdi_label':      label,
                        'fdi_confidence': f'{conf:.4f}',
                        'fdi_detected':   det,
                        't_fault_s':      f'{t_f:.4f}',
                        't_detect_s':     f'{t_d:.4f}' if t_d else '',
                        'gyro_x':         f'{imu[0]:.5f}',
                        'gyro_y':         f'{imu[1]:.5f}',
                        'gyro_z':         f'{imu[2]:.5f}',
                        'acc_x':          f'{imu[3]:.5f}',
                        'acc_y':          f'{imu[4]:.5f}',
                        'acc_z':          f'{imu[5]:.5f}',
                        'm1_norm':        f'{m1:.4f}',
                        'm2_norm':        f'{m2:.4f}',
                        'm3_norm':        f'{m3:.4f}',
                        'm4_norm':        f'{m4:.4f}',
                        'roll_deg':       f'{roll:.4f}',
                        'pitch_deg':      f'{pitch:.4f}',
                        'yaw_deg':        f'{data.get("stabilizer.yaw", 0.0):.4f}',
                        'z_m':            f'{data.get("stateEstimate.z", 0.0):.4f}',
                        'ft_active':      data.get('ftAlloc.active', 0),
                        'ft_failed_motor':data.get('ftAlloc.failedMotor', 0),
                        'ft_residual_yaw':f'{data.get("ftAlloc.residualYaw", 0.0):.5f}',
                    })

                cf.log.add_config(lc_imu)
                cf.log.add_config(lc_state)
                lc_imu.data_received_cb.add_callback(imu_cb)
                lc_state.data_received_cb.add_callback(state_cb)
                safety.start()

                try:
                    lc_imu.start()
                    lc_state.start()
                    ok = self._fly(cf, safety)
                finally:
                    lc_imu.stop()
                    lc_state.stop()
                    safety.stop()
                    try:
                        cf.param.set_value('powerDist.failedMotor', '0')
                    except Exception:
                        pass

        except Exception as exc:
            print(f'[SITL] ERROR: {exc}')
            return None

        return self._save_csv(f'e2e_sitl_motor{self.fault_motor}.csv')

    # ── flight sequence ───────────────────────────────────────────────────────

    def _fly(self, cf, safety) -> bool:
        dt = LOG_PERIOD_MS / 1000.0

        # takeoff ramp
        print('[SITL] Taking off …')
        with self._lock:
            self._phase = Phase.TAKEOFF
        steps = max(1, int(2.0 / dt))
        for i in range(steps + 1):
            if safety.killed:
                return False
            z = HOVER_HEIGHT_M * i / steps
            cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, z)
            time.sleep(dt)

        # stabilise hover
        with self._lock:
            self._phase = Phase.HOVER
        print(f'[SITL] Stabilising {STAB_WAIT_S:.0f} s …')
        t0 = time.monotonic()
        while time.monotonic() - t0 < STAB_WAIT_S:
            if safety.killed:
                return False
            cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, HOVER_HEIGHT_M)
            time.sleep(dt)

        # inject fault — kill motor externally
        t_fault_wall = time.monotonic()
        with self._lock:
            self._t_fault_s = t_fault_wall - self._t_start
            self._phase     = Phase.FAULT_BLIND
        print(f'[SITL] Injecting fault: powerDist.failedMotor = {self.fault_motor}')
        cf.param.set_value('powerDist.failedMotor', str(self.fault_motor))
        time.sleep(0.05)

        # wait for FDI to detect (or timeout)
        print('[SITL] Waiting for FDI detection …')
        detect_timeout = 8.0
        t0 = time.monotonic()
        while time.monotonic() - t0 < detect_timeout:
            if safety.killed:
                return False
            cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, HOVER_HEIGHT_M)
            with self._lock:
                det = self._fdi_detected
            if det:
                break
            time.sleep(dt)
        else:
            print('[SITL] WARNING: FDI detection timed out — proceeding anyway.')

        # FT recovery hold
        with self._lock:
            self._phase = Phase.FT_RECOVERY
        print(f'[SITL] FT recovery hold {FT_HOLD_S:.1f} s …')
        t0 = time.monotonic()
        while time.monotonic() - t0 < FT_HOLD_S:
            if safety.killed:
                return False
            cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, HOVER_HEIGHT_M)
            time.sleep(dt)

        # descent
        with self._lock:
            self._phase = Phase.DESCENT
        print(f'[SITL] Descending at {DESCENT_RATE_MS} m/s …')
        current_z = HOVER_HEIGHT_M
        while current_z > LAND_THRESHOLD_M and not safety.killed:
            current_z = max(LAND_THRESHOLD_M, current_z - DESCENT_RATE_MS * dt)
            cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, current_z)
            time.sleep(dt)

        # land
        with self._lock:
            self._phase = Phase.LANDED
        print('[SITL] Landed — cutting motors.')
        for _ in range(10):
            cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, 0.0)
            time.sleep(0.02)
        cf.commander.send_stop_setpoint()
        time.sleep(0.3)
        return True

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _wait_for_estimator(scf, SyncLogger, LogConfig, timeout_s=30.0):
        print('[SITL] Waiting for Kalman estimator …', end='', flush=True)
        lc = LogConfig('_KF', period_in_ms=100)
        lc.add_variable('stateEstimate.z', 'float')
        t0 = time.time()
        with SyncLogger(scf, lc) as sl:
            for _, _, data in sl:
                z = data.get('stateEstimate.z', 0.0)
                if z > 0.02 or time.time() - t0 > timeout_s:
                    break
        print(f' ready (z={z:.3f} m)')

    @staticmethod
    def _build_log(cf, var_list, name, period_ms):
        from cflib.crazyflie.log import LogConfig
        toc = cf.log.toc.toc if hasattr(cf.log.toc, 'toc') else {}
        lc  = LogConfig(name, period_in_ms=period_ms)
        skipped = []
        for var_name, var_type in var_list:
            group, var = var_name.split('.')
            try:
                _ = toc[group][var]
                lc.add_variable(var_name, var_type)
            except (KeyError, AttributeError):
                skipped.append(var_name)
        if skipped:
            print(f'[SITL] Skipped (not in TOC): {skipped}')
        return lc

    def _save_csv(self, filename: str) -> Optional[Path]:
        if not self._rows:
            print('[SITL] No rows to save.')
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        with open(path, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=E2E_FIELDNAMES,
                                    extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self._rows)
        print(f'[SITL] Saved {len(self._rows)} rows → {path}')
        return path


# ══════════════════════════════════════════════════════════════════════════════
# SITL safety monitor (copied locally to keep script self-contained)
# ══════════════════════════════════════════════════════════════════════════════

class _SafetyMonitor:
    def __init__(self, cf, threshold_deg=SAFETY_ANGLE_DEG):
        self._cf        = cf
        self._threshold = threshold_deg
        self._roll      = 0.0
        self._pitch     = 0.0
        self._lock      = threading.Lock()
        self._killed    = False
        self._stop_evt  = threading.Event()

    def update_attitude(self, roll, pitch):
        with self._lock:
            self._roll, self._pitch = roll, pitch

    @property
    def killed(self):
        return self._killed

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name='SafetyMon')
        self._thread.start()

    def stop(self):
        self._stop_evt.set()

    def _run(self):
        while not self._stop_evt.is_set():
            with self._lock:
                roll, pitch = self._roll, self._pitch
            if abs(roll) > self._threshold or abs(pitch) > self._threshold:
                print(f'\n[SAFETY] roll={roll:.1f}° pitch={pitch:.1f}° — cutting motors!')
                self._cf.commander.send_stop_setpoint()
                self._killed = True
                self._stop_evt.set()
                return
            time.sleep(0.02)


# ══════════════════════════════════════════════════════════════════════════════
# MuJoCo backend
# ══════════════════════════════════════════════════════════════════════════════

# ── Physical constants (CF2, match mujoco_ft_test.py) ─────────────────────────
MASS           = 0.027
G_N            = 9.81
HOVER_F        = MASS * G_N
ARM_LEN        = 0.046
ARM            = ARM_LEN * math.sin(math.radians(45))
THRUST2TORQUE  = 0.005964552
MAX_MOTOR_F    = 0.60 / 4
SPIN           = np.array([1.0, -1.0, 1.0, -1.0])
Ixx            = 2.3951e-5
Izz            = 3.2347e-5
MJ_DT          = 0.002     # s — 500 Hz

RGBA_OK   = np.array([0.20, 0.80, 0.20, 1.0])
RGBA_FAIL = np.array([0.90, 0.10, 0.10, 1.0])


def _alloc_4motor(T, r, p, y):
    return np.clip([T-r+p+y, T-r-p-y, T+r-p+y, T+r+p-y], 0.0, MAX_MOTOR_F)


def _alloc_ft(T, r, p, failed):
    m = np.zeros(4)
    if   failed == 1: m[1]=2*(T-r); m[2]=2*(r-p); m[3]=2*(T+p); yN=-T+r-p
    elif failed == 2: m[0]=2*(T-r); m[2]=2*(T-p); m[3]=2*(r+p); yN=T-r-p
    elif failed == 3: m[0]=2*(p-r); m[1]=2*(T-p); m[3]=2*(T+r); yN=p-r-T
    elif failed == 4: m[0]=2*(T+p); m[1]=-2*(r+p);m[2]=2*(T+r); yN=T+r+p
    else:             return _alloc_4motor(T,r,p,0), 0.0
    return np.clip(m,0,MAX_MOTOR_F), THRUST2TORQUE*4*yN


def _motors_to_wrench(f):
    thrust    = float(f.sum())
    tau_roll  = ARM * float((f[2]+f[3]) - (f[0]+f[1]))
    tau_pitch = ARM * float((f[0]+f[3]) - (f[1]+f[2]))
    tau_yaw   = THRUST2TORQUE * float(SPIN @ f)
    return thrust, tau_roll, tau_pitch, tau_yaw


@dataclass
class _HoverCtrl:
    target_z: float = HOVER_HEIGHT_M
    kp_z:  float = field(default_factory=lambda: MASS*25.0)
    ki_z:  float = field(default_factory=lambda: MASS*4.0)
    kd_z:  float = field(default_factory=lambda: MASS*10.0)
    kp_rp: float = field(default_factory=lambda: Ixx*400.0)
    kd_rp: float = field(default_factory=lambda: Ixx*40.0)
    kp_yaw:float = field(default_factory=lambda: Izz*25.0)
    kd_yaw:float = field(default_factory=lambda: Izz*10.0)
    _zi:   float = field(default=0.0, init=False, repr=False)

    def reset(self):
        self._zi = 0.0

    def compute(self, st, dt):
        z   = float(st['pos'][2])
        vz  = float(st['vel'][2])
        roll,pitch,yaw = st['rpy']
        ox,oy,oz       = st['gyro']
        ze             = self.target_z - z
        self._zi       = float(np.clip(self._zi + ze*dt, -0.5, 0.5))
        Ft = float(np.clip(HOVER_F + self.kp_z*ze + self.ki_z*self._zi - self.kd_z*vz,
                           0.0, MAX_MOTOR_F*4))
        tr = -self.kp_rp*roll  - self.kd_rp*ox
        tp = -self.kp_rp*pitch - self.kd_rp*oy
        ty = -self.kp_yaw*yaw  - self.kd_yaw*oz
        return Ft/4, tr/(4*ARM), tp/(4*ARM), ty/(4*THRUST2TORQUE)


def _mj_get_state(data):
    import mujoco
    quat = data.sensor('body_quat').data.copy()
    gyro = data.sensor('body_gyro').data.copy()
    pos  = data.sensor('body_pos').data.copy()
    vel  = data.sensor('body_linvel').data.copy()
    w,x,y,z = quat
    roll  = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = math.asin(float(np.clip(2*(w*y-z*x), -1, 1)))
    yaw   = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return {'pos':pos,'vel':vel,'rpy':np.array([roll,pitch,yaw]),'gyro':gyro}


def _mj_apply_wrench(model, data, thrust, tau_roll, tau_pitch, tau_yaw):
    import mujoco
    bid  = model.body('cf2').id
    xmat = data.xmat[bid].reshape(3,3)
    data.xfrc_applied[bid,:3] = xmat @ np.array([0,0,thrust])
    data.xfrc_applied[bid,3:] = xmat @ np.array([tau_roll,tau_pitch,tau_yaw])


def _fdi_mujoco_mock(sim_time, fault_time, fault_injected, fault_motor,
                     detect_lag=DETECT_LAG_S, threshold=CONFIDENCE_THRESH):
    """
    Sim-time-aware FDI mock for MuJoCo.

    Returns (label, confidence).
    Before fault injection or during blind window: returns (0, high_healthy_conf).
    As sim_time approaches fault_time+detect_lag, confidence in fault class rises
    smoothly.  After detect_lag, returns (fault_motor, 0.92).
    """
    if not fault_injected:
        return 0, 0.97

    elapsed = sim_time - fault_time
    if elapsed < 0:
        return 0, 0.97

    if elapsed >= detect_lag:
        return fault_motor, 0.92

    # Smooth ramp: healthy conf drops, fault conf rises
    frac         = elapsed / detect_lag               # 0 → 1
    fault_conf   = 0.97 * frac                        # 0 → 0.92…
    healthy_conf = 0.97 * (1.0 - 0.85 * frac)        # 0.97 → ~0.15
    if fault_conf > healthy_conf:
        return fault_motor, fault_conf
    return 0, healthy_conf


class MuJoCoBackend:
    """
    MuJoCo end-to-end test.

    At FAULT_TIME_S the failed motor's force is zeroed in the physics.
    The 4-motor allocator continues running during the blind window, so the
    vehicle becomes unstable.  Once the FDI mock declares the fault, the
    3-motor FT allocator takes over.
    """

    def __init__(
        self,
        xml_path:     Path  = MJ_XML_PATH,
        fault_motor:  int   = 2,
        detect_lag_s: float = DETECT_LAG_S,
        threshold:    float = CONFIDENCE_THRESH,
        headless:     bool  = True,
        output_dir:   Path  = RESULTS_DIR,
    ):
        self.xml_path     = Path(xml_path)
        self.fault_motor  = fault_motor
        self.detect_lag_s = detect_lag_s
        self.threshold    = threshold
        self.headless     = headless
        self.output_dir   = Path(output_dir)

    def run(self) -> Optional[Path]:
        try:
            import mujoco
            import mujoco.viewer
        except ImportError:
            print('[MuJoCo] mujoco not installed — pip install mujoco')
            return None

        xml = self.xml_path.expanduser().resolve()
        if not xml.exists():
            print(f'[MuJoCo] XML not found: {xml}')
            return None

        model = mujoco.MjModel.from_xml_path(str(xml))
        data  = mujoco.MjData(model)
        for i in range(1, 5):
            model.geom_rgba[model.geom(f'prop{i}').id] = RGBA_OK

        if self.headless:
            rows = self._sim_loop(model, data, viewer=None)
        else:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                viewer.cam.distance  = 2.0
                viewer.cam.elevation = -25
                viewer.cam.azimuth   = 135
                rows = self._sim_loop(model, data, viewer=viewer)

        return self._save_csv(rows, f'e2e_mujoco_motor{self.fault_motor}.csv')

    def _sim_loop(self, model, data, viewer) -> list[dict]:
        import mujoco
        mujoco.mj_resetData(model, data)
        ctrl    = _HoverCtrl(target_z=HOVER_HEIGHT_M)
        ctrl.reset()
        dt      = float(model.opt.timestep)

        phase           = Phase.HOVER
        fault_injected  = False
        fdi_detected    = False
        t_fault         = 0.0
        t_detect        = 0.0
        descent_z       = HOVER_HEIGHT_M
        prev_vel        = None
        rows: list[dict] = []
        TOTAL_TIMEOUT   = FAULT_TIME_S + 3.0 + 30.0   # generous upper bound

        print(f'[MuJoCo] Simulating … fault at t={FAULT_TIME_S:.1f}s '
              f'(motor {self.fault_motor})  detect_lag={self.detect_lag_s:.2f}s')

        while data.time < TOTAL_TIMEOUT:
            t  = data.time
            st = _mj_get_state(data)
            z  = float(st['pos'][2])

            # ── compute body-frame acceleration ───────────────────────────
            if prev_vel is None:
                acc_body = np.array([0.0, 0.0, G_N])
            else:
                acc_world = (st['vel'] - prev_vel) / dt
                bid  = model.body('cf2').id
                xmat = data.xmat[bid].reshape(3, 3)
                # body = R^T @ world
                acc_body = xmat.T @ acc_world
                # sensor acc includes gravity (down ≈ -9.81 in world z):
                acc_body[2] += G_N   # add gravity contribution (approx)
            prev_vel = st['vel'].copy()

            imu_sample = list(st['gyro']) + list(acc_body)

            # ── phase transitions ─────────────────────────────────────────
            if not fault_injected and t >= FAULT_TIME_S:
                fault_injected = True
                t_fault        = t
                phase          = Phase.FAULT_BLIND
                model.geom_rgba[model.geom(f'prop{self.fault_motor}').id] = RGBA_FAIL
                print(f'[MuJoCo] t={t:.2f}s  FAULT INJECTED  (motor {self.fault_motor})'
                      f'  — running 4-motor alloc during blind window')

            if fault_injected and not fdi_detected:
                label, conf = _fdi_mujoco_mock(
                    t, t_fault, fault_injected, self.fault_motor,
                    self.detect_lag_s, self.threshold)
                if label != 0 and conf >= self.threshold:
                    fdi_detected = True
                    t_detect     = t
                    phase        = Phase.FT_RECOVERY
                    print(f'[MuJoCo] t={t:.2f}s  FDI DETECTED motor {label}'
                          f'  conf={conf:.3f}'
                          f'  latency={(t_detect-t_fault)*1000:.0f} ms')
            else:
                label = self.fault_motor if fdi_detected else 0
                conf  = 0.92 if fdi_detected else 0.97

            if phase == Phase.FT_RECOVERY and t >= t_detect + FT_HOLD_S:
                phase     = Phase.DESCENT
                descent_z = HOVER_HEIGHT_M
                print(f'[MuJoCo] t={t:.2f}s  BEGIN DESCENT')

            if phase == Phase.DESCENT:
                descent_z      = max(0.0, descent_z - DESCENT_RATE_MS * dt)
                ctrl.target_z  = descent_z
                if z < LAND_THRESHOLD_M:
                    phase = Phase.LANDED
                    print(f'[MuJoCo] t={t:.2f}s  LANDED  (z={z:.3f} m)')

            if phase == Phase.LANDED:
                data.xfrc_applied[:] = 0.0
                mujoco.mj_step(model, data)
                if viewer is not None:
                    viewer.sync()
                break

            # ── controller ────────────────────────────────────────────────
            if phase in (Phase.HOVER, Phase.FT_RECOVERY):
                ctrl.target_z = HOVER_HEIGHT_M

            T, r, p, yc = ctrl.compute(st, dt)

            # ── allocator ─────────────────────────────────────────────────
            if phase == Phase.FAULT_BLIND:
                # still on 4-motor — motor 2 is physically dead → instability
                f4 = _alloc_4motor(T, r, p, yc)
                f4[self.fault_motor - 1] = 0.0   # physical motor dead
                thrust, tr, tp, ty = _motors_to_wrench(f4)
                ft_res = 0.0
                forces = f4
            elif fdi_detected:
                forces, ft_res = _alloc_ft(T, r, p, self.fault_motor)
                thrust, tr, tp, ty = _motors_to_wrench(forces)
            else:
                forces = _alloc_4motor(T, r, p, yc)
                thrust, tr, tp, ty = _motors_to_wrench(forces)
                ft_res = 0.0

            _mj_apply_wrench(model, data, thrust, tr, tp, ty)

            # ── log row ───────────────────────────────────────────────────
            rpy = st['rpy']
            rows.append({
                'backend':        'mujoco',
                't_s':            f'{t:.4f}',
                'phase':          phase.value,
                'fdi_label':      label,
                'fdi_confidence': f'{conf:.4f}',
                'fdi_detected':   fdi_detected,
                't_fault_s':      f'{t_fault:.4f}',
                't_detect_s':     f'{t_detect:.4f}' if fdi_detected else '',
                'gyro_x':         f'{imu_sample[0]:.5f}',
                'gyro_y':         f'{imu_sample[1]:.5f}',
                'gyro_z':         f'{imu_sample[2]:.5f}',
                'acc_x':          f'{imu_sample[3]:.5f}',
                'acc_y':          f'{imu_sample[4]:.5f}',
                'acc_z':          f'{imu_sample[5]:.5f}',
                'm1_norm':        f'{forces[0]/MAX_MOTOR_F:.4f}',
                'm2_norm':        f'{forces[1]/MAX_MOTOR_F:.4f}',
                'm3_norm':        f'{forces[2]/MAX_MOTOR_F:.4f}',
                'm4_norm':        f'{forces[3]/MAX_MOTOR_F:.4f}',
                'roll_deg':       f'{math.degrees(rpy[0]):.4f}',
                'pitch_deg':      f'{math.degrees(rpy[1]):.4f}',
                'yaw_deg':        f'{math.degrees(rpy[2]):.4f}',
                'z_m':            f'{z:.4f}',
                'ft_active':      1 if fdi_detected else 0,
                'ft_failed_motor':self.fault_motor if fdi_detected else 0,
                'ft_residual_yaw':f'{ft_res:.5f}',
            })

            mujoco.mj_step(model, data)
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.cam.lookat[:] = st['pos']
                if int(t / dt) % 5 == 0:
                    viewer.sync()

        return rows

    def _save_csv(self, rows, filename) -> Optional[Path]:
        if not rows:
            print('[MuJoCo] No rows to save.')
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        with open(path, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=E2E_FIELDNAMES,
                                    extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        print(f'[MuJoCo] Saved {len(rows)} rows → {path}')
        return path


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

MOTOR_COLORS   = ['#e6194b','#3cb44b','#4363d8','#f58231']
PHASE_COLORS   = {
    Phase.HOVER.value:       '#d0e8ff',
    Phase.FAULT_BLIND.value: '#ffe0c0',
    Phase.FT_RECOVERY.value: '#d0ffd0',
    Phase.DESCENT.value:     '#f0f0f0',
    Phase.LANDED.value:      '#e0e0e0',
    Phase.TAKEOFF.value:     '#e8e8ff',
}


def _load_csv(path: Path) -> dict:
    """Load e2e CSV into dict of numpy arrays."""
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f'Empty CSV: {path}')

    def _f(k, default=0.0):
        return np.array([float(r.get(k, default) or default) for r in rows])

    def _b(k):
        return np.array([r.get(k,'False') in ('True','1','true') for r in rows])

    t       = _f('t_s')
    t      -= t[0]   # normalise to start at 0

    t_fault_raw = _f('t_fault_s')
    t_fault = float(t_fault_raw[t_fault_raw > 0][0]) if (t_fault_raw > 0).any() else None

    t_det_raw  = _f('t_detect_s')
    t_detect   = float(t_det_raw[t_det_raw > 0][0]) if (t_det_raw > 0).any() else None

    return {
        'path':       path,
        'backend':    rows[0].get('backend','?'),
        't':          t,
        't_fault':    t_fault,
        't_detect':   t_detect,
        'phase':      [r.get('phase','') for r in rows],
        'fdi_label':  _f('fdi_label'),
        'fdi_conf':   _f('fdi_confidence'),
        'fdi_det':    _b('fdi_detected'),
        'm1':         _f('m1_norm'),
        'm2':         _f('m2_norm'),
        'm3':         _f('m3_norm'),
        'm4':         _f('m4_norm'),
        'roll':       _f('roll_deg'),
        'pitch':      _f('pitch_deg'),
        'yaw':        _f('yaw_deg'),
        'z':          _f('z_m'),
        'ft_active':  _f('ft_active'),
        'gyro_x':     _f('gyro_x'),
        'gyro_y':     _f('gyro_y'),
        'gyro_z':     _f('gyro_z'),
    }


def _shade_phases(ax, t, phases):
    """Draw translucent phase background spans."""
    if not phases:
        return
    prev_phase = phases[0]
    t_start    = t[0]
    for i in range(1, len(t)):
        if phases[i] != prev_phase or i == len(t) - 1:
            color = PHASE_COLORS.get(prev_phase, '#ffffff')
            ax.axvspan(t_start, t[i], alpha=0.25, color=color,
                       zorder=0, label='_nolegend_')
            t_start    = t[i]
            prev_phase = phases[i]


def plot_timeline(d: dict, save_path: Path, fault_motor: int) -> None:
    """
    5-panel timeline figure for one backend's e2e run.

    Panels:
      1. Altitude z with fault/detect markers
      2. Roll & pitch
      3. Motor commands (normalised)
      4. FDI confidence + label
      5. Gyroscope magnitude (proxy for anomaly)
    """
    t        = d['t']
    t_fault  = d['t_fault']
    t_detect = d['t_detect']
    backend  = d['backend'].upper()
    lat_ms   = f'{(t_detect - t_fault)*1000:.0f} ms' if (t_fault and t_detect) else 'N/A'

    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(
        f'End-to-End Gamma FDI + FT Landing — {backend}  '
        f'(Motor {fault_motor} failed)\n'
        f'Detection latency: {lat_ms}',
        fontsize=13, fontweight='bold', y=0.99,
    )

    vkw_f = dict(color='#cc0000', lw=1.6, ls='--', zorder=5)
    vkw_d = dict(color='#007700', lw=1.6, ls=':',  zorder=5)

    def _add_phase_bg(ax):
        _shade_phases(ax, t, d['phase'])

    def _add_markers(ax):
        if t_fault  is not None:
            ax.axvline(t_fault,  **vkw_f, label=f'Fault inject (t={t_fault:.1f}s)')
        if t_detect is not None:
            ax.axvline(t_detect, **vkw_d, label=f'FDI detect (t={t_detect:.1f}s)')

    def _fmt(ax, ylabel, ylim=None, legend=True, grid=True):
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        if ylim:
            ax.set_ylim(*ylim)
        if grid:
            ax.grid(True, alpha=0.3, lw=0.5)
        if legend:
            ax.legend(fontsize=7, loc='upper right', ncol=2)

    # ── Panel 1: Altitude ──────────────────────────────────────────────────
    ax = axes[0]
    _add_phase_bg(ax)
    ax.plot(t, d['z'], color='#2166ac', lw=1.6, label='Altitude z')
    ax.axhline(HOVER_HEIGHT_M, color='gray', lw=0.8, ls=':', label=f'Target {HOVER_HEIGHT_M} m')
    ax.axhline(0.0,            color='k',    lw=0.5, ls='-', alpha=0.4)
    _add_markers(ax)
    _fmt(ax, 'Altitude z [m]', ylim=(-0.05, 0.75))
    ax.set_title('Altitude', fontsize=10, pad=2)

    # Annotate detection latency
    if t_fault and t_detect:
        mid = (t_fault + t_detect) / 2
        ax.annotate(
            f'Δt={lat_ms}',
            xy=(t_detect, HOVER_HEIGHT_M * 0.85),
            xytext=(mid, HOVER_HEIGHT_M * 0.95),
            arrowprops=dict(arrowstyle='->', color='green', lw=1.2),
            fontsize=8, color='green', ha='center',
        )

    # ── Panel 2: Attitude ──────────────────────────────────────────────────
    ax = axes[1]
    _add_phase_bg(ax)
    ax.plot(t, d['roll'],  color='#e31a1c', lw=1.4, label='Roll')
    ax.plot(t, d['pitch'], color='#1f78b4', lw=1.4, label='Pitch')
    ax.axhline(0, color='k', lw=0.5, ls='-', alpha=0.4)
    ax.axhline( SAFETY_ANGLE_DEG, color='#ff7700', lw=0.8, ls=':', alpha=0.6, label=f'Safety ±{SAFETY_ANGLE_DEG:.0f}°')
    ax.axhline(-SAFETY_ANGLE_DEG, color='#ff7700', lw=0.8, ls=':', alpha=0.6)
    _add_markers(ax)
    _fmt(ax, 'Angle [°]', ylim=(-65, 65))
    ax.set_title('Roll & Pitch', fontsize=10, pad=2)

    # ── Panel 3: Motor commands ────────────────────────────────────────────
    ax = axes[2]
    _add_phase_bg(ax)
    labels = ['M1 (FL,CCW)','M2 (BL,CW)','M3 (BR,CCW)','M4 (FR,CW)']
    for i, (key, lbl) in enumerate(zip(['m1','m2','m3','m4'], labels)):
        dead  = (i + 1) == fault_motor
        ax.plot(t, d[key],
                color=MOTOR_COLORS[i], lw=0.7 if dead else 1.4,
                ls=':' if dead else '-', label=lbl, alpha=0.85)
    ax.axhline(0, color='k', lw=0.5, alpha=0.4)
    _add_markers(ax)
    _fmt(ax, 'Motor thrust (%)', ylim=(-0.05, 1.15))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title('Motor Commands', fontsize=10, pad=2)

    # ── Panel 4: FDI output ────────────────────────────────────────────────
    ax = axes[3]
    _add_phase_bg(ax)
    ax.plot(t, d['fdi_conf'],   color='#6a3d9a', lw=1.5, label='FDI confidence')
    ax.fill_between(t, d['fdi_conf'], alpha=0.15, color='#6a3d9a')
    ax.axhline(CONFIDENCE_THRESH, color='#ff7700', lw=1.2, ls='--',
               label=f'Threshold {CONFIDENCE_THRESH:.0%}')
    # shade detection region
    if t_detect is not None:
        ax.axvspan(t_detect, t[-1], alpha=0.10, color='green', label='FDI active')
    # FDI label on secondary y
    ax2 = ax.twinx()
    ax2.step(t, d['fdi_label'], color='#ff7f00', lw=1.0, where='post', alpha=0.7)
    ax2.set_yticks([0,1,2,3,4])
    ax2.set_yticklabels(['healthy','M1','M2','M3','M4'], fontsize=7)
    ax2.set_ylim(-0.5, 5.5)
    ax2.tick_params(labelsize=7)
    _add_markers(ax)
    _fmt(ax, 'Confidence', ylim=(-0.05, 1.10))
    ax.set_title('Gamma FDI Output', fontsize=10, pad=2)

    # ── Panel 5: Gyro magnitude ────────────────────────────────────────────
    ax = axes[4]
    _add_phase_bg(ax)
    gyro_mag = np.sqrt(d['gyro_x']**2 + d['gyro_y']**2 + d['gyro_z']**2)
    ax.plot(t, gyro_mag, color='#b2182b', lw=1.2, label='|ω| (rad/s)')
    ax.fill_between(t, gyro_mag, alpha=0.12, color='#b2182b')
    _add_markers(ax)
    _fmt(ax, '|ω| [rad/s]', legend=True)
    ax.set_xlabel('Time [s]', fontsize=9)
    ax.set_title('Gyro Magnitude (IMU Anomaly Indicator)', fontsize=10, pad=2)

    # ── Phase legend ───────────────────────────────────────────────────────
    phase_patches = [
        mpatches.Patch(color=PHASE_COLORS.get(p.value, 'white'), alpha=0.5, label=p.value)
        for p in [Phase.HOVER, Phase.FAULT_BLIND, Phase.FT_RECOVERY, Phase.DESCENT]
    ]
    fig.legend(handles=phase_patches, loc='lower center', ncol=4,
               fontsize=8, title='Flight Phase', title_fontsize=8,
               bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Plot] Saved: {save_path.name}')


def plot_comparison(sitl: dict, mujoco: dict, save_path: Path,
                    fault_motor: int) -> None:
    """
    Side-by-side comparison: SITL (left) vs MuJoCo (right).
    Rows: altitude, attitude, FDI confidence, gyro magnitude.
    """
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex='col')
    fig.suptitle(
        f'SITL vs MuJoCo — End-to-End Gamma FDI (Motor {fault_motor} failed)',
        fontsize=13, fontweight='bold',
    )

    backends = [sitl, mujoco]
    titles   = ['CrazySim SITL', 'MuJoCo (500 Hz)']

    for col, (d, title) in enumerate(zip(backends, titles)):
        t        = d['t']
        t_fault  = d['t_fault']
        t_detect = d['t_detect']
        lat_ms   = f'{(t_detect-t_fault)*1000:.0f} ms' if (t_fault and t_detect) else 'N/A'

        vkw_f = dict(color='#cc0000', lw=1.4, ls='--')
        vkw_d = dict(color='#007700', lw=1.4, ls=':')

        # Row 0: Altitude
        ax = axes[0][col]
        ax.plot(t, d['z'], lw=1.5, color='#2166ac')
        ax.axhline(HOVER_HEIGHT_M, color='gray', lw=0.8, ls=':')
        if t_fault:  ax.axvline(t_fault,  **vkw_f)
        if t_detect: ax.axvline(t_detect, **vkw_d)
        _shade_phases(ax, t, d['phase'])
        ax.set_ylim(-0.05, 0.75)
        ax.set_ylabel('z [m]', fontsize=9)
        ax.set_title(f'{title}\nDetect latency: {lat_ms}', fontsize=10)
        ax.grid(True, alpha=0.3, lw=0.5)

        # Row 1: Attitude
        ax = axes[1][col]
        ax.plot(t, d['roll'],  color='#e31a1c', lw=1.2, label='Roll')
        ax.plot(t, d['pitch'], color='#1f78b4', lw=1.2, label='Pitch')
        ax.axhline(0, color='k', lw=0.4, alpha=0.4)
        if t_fault:  ax.axvline(t_fault,  **vkw_f)
        if t_detect: ax.axvline(t_detect, **vkw_d)
        _shade_phases(ax, t, d['phase'])
        ax.set_ylim(-65, 65)
        ax.set_ylabel('Angle [°]', fontsize=9)
        if col == 0: ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, lw=0.5)

        # Row 2: FDI confidence
        ax = axes[2][col]
        ax.plot(t, d['fdi_conf'], color='#6a3d9a', lw=1.3)
        ax.fill_between(t, d['fdi_conf'], alpha=0.15, color='#6a3d9a')
        ax.axhline(CONFIDENCE_THRESH, color='#ff7700', lw=1.1, ls='--',
                   label=f'Threshold {CONFIDENCE_THRESH:.0%}')
        if t_fault:  ax.axvline(t_fault,  **vkw_f, label='Fault')
        if t_detect: ax.axvline(t_detect, **vkw_d, label='Detect')
        ax.set_ylim(-0.05, 1.10)
        ax.set_ylabel('FDI confidence', fontsize=9)
        if col == 0: ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, lw=0.5)

        # Row 3: Gyro magnitude
        ax = axes[3][col]
        gyro_mag = np.sqrt(d['gyro_x']**2 + d['gyro_y']**2 + d['gyro_z']**2)
        ax.plot(t, gyro_mag, color='#b2182b', lw=1.1)
        ax.fill_between(t, gyro_mag, alpha=0.12, color='#b2182b')
        if t_fault:  ax.axvline(t_fault,  **vkw_f)
        if t_detect: ax.axvline(t_detect, **vkw_d)
        ax.set_ylabel('|ω| [rad/s]', fontsize=9)
        ax.set_xlabel('Time [s]', fontsize=9)
        ax.grid(True, alpha=0.3, lw=0.5)

        for ax in axes[:, col]:
            ax.tick_params(labelsize=8)

    # shared legend for phase colours
    phase_patches = [
        mpatches.Patch(color=PHASE_COLORS.get(p.value,'white'), alpha=0.5, label=p.value)
        for p in [Phase.HOVER, Phase.FAULT_BLIND, Phase.FT_RECOVERY, Phase.DESCENT]
    ]
    fig.legend(handles=phase_patches, loc='lower center', ncol=4, fontsize=8,
               title='Flight Phase', title_fontsize=8,
               bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Plot] Saved: {save_path.name}')


def _print_summary(label: str, d: dict, fault_motor: int) -> None:
    """Print detection latency, max attitude deviation, landing time."""
    t_f = d.get('t_fault')
    t_d = d.get('t_detect')
    lat = f'{(t_d-t_f)*1000:.0f} ms' if (t_f is not None and t_d is not None) else 'N/A'

    ft_mask = d['ft_active'] > 0
    roll_ft  = np.abs(d['roll'][ft_mask])  if ft_mask.any() else np.array([0.0])
    pitch_ft = np.abs(d['pitch'][ft_mask]) if ft_mask.any() else np.array([0.0])

    z_hover_mask = np.array([p == Phase.HOVER.value for p in d['phase']])
    z_hover_mean = d['z'][z_hover_mask].mean() if z_hover_mask.any() else float('nan')

    print(f'\n  [{label}]')
    print(f'    Fault motor       : {fault_motor}')
    print(f'    Fault injected at : {t_f:.3f} s' if t_f else '    Fault time: N/A')
    print(f'    FDI detection at  : {t_d:.3f} s' if t_d else '    FDI detect: N/A')
    print(f'    Detection latency : {lat}')
    print(f'    Hover z (mean)    : {z_hover_mean:.3f} m')
    print(f'    FT max |roll|     : {roll_ft.max():.2f}°')
    print(f'    FT max |pitch|    : {pitch_ft.max():.2f}°')
    print(f'    Land time         : {d["t"][-1]:.2f} s')
    print(f'    Rows logged       : {len(d["t"])}')


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description='End-to-end Gamma FDI + fault-tolerant landing test',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--backend', choices=['sitl', 'mujoco', 'both'], default='both',
        help='Which simulation backend(s) to run',
    )
    parser.add_argument(
        '--uri', default=DEFAULT_URI,
        help='CrazySim SITL URI (SITL only)',
    )
    parser.add_argument(
        '--xml', type=Path, default=MJ_XML_PATH,
        help='MuJoCo MJCF XML path',
    )
    parser.add_argument(
        '--fault-motor', type=int, default=2, choices=[1,2,3,4],
        help='Which motor to kill',
    )
    parser.add_argument(
        '--detect-lag', type=float, default=DETECT_LAG_S,
        help='FDI detection latency [s] after fault injection',
    )
    parser.add_argument(
        '--threshold', type=float, default=CONFIDENCE_THRESH,
        help='FDI confidence threshold',
    )
    parser.add_argument(
        '--headless', action='store_true',
        help='MuJoCo headless (no viewer)',
    )
    parser.add_argument(
        '--output-dir', type=Path, default=RESULTS_DIR,
        help='Directory for CSV and PNG outputs',
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help='Skip plotting (useful for CI)',
    )
    args = parser.parse_args()

    out_dir      = args.output_dir
    fault_motor  = args.fault_motor

    print('=' * 64)
    print('End-to-End Gamma FDI + Fault-Tolerant Landing Test')
    print('=' * 64)
    print(f'  Backend     : {args.backend}')
    print(f'  Fault motor : {fault_motor}')
    print(f'  Detect lag  : {args.detect_lag:.2f} s')
    print(f'  Threshold   : {args.threshold:.0%}')
    print(f'  Output dir  : {out_dir}')
    print()

    sitl_csv   = None
    mujoco_csv = None

    # ── SITL backend ───────────────────────────────────────────────────────
    if args.backend in ('sitl', 'both'):
        print('─' * 64)
        print('SITL BACKEND')
        print('─' * 64)
        backend = SITLBackend(
            uri=args.uri,
            fault_motor=fault_motor,
            detect_lag_s=args.detect_lag,
            threshold=args.threshold,
            output_dir=out_dir,
        )
        sitl_csv = backend.run()

    # ── MuJoCo backend ────────────────────────────────────────────────────
    if args.backend in ('mujoco', 'both'):
        print('─' * 64)
        print('MUJOCO BACKEND')
        print('─' * 64)
        backend = MuJoCoBackend(
            xml_path=args.xml,
            fault_motor=fault_motor,
            detect_lag_s=args.detect_lag,
            threshold=args.threshold,
            headless=args.headless,
            output_dir=out_dir,
        )
        mujoco_csv = backend.run()

    # ── plotting ───────────────────────────────────────────────────────────
    if not args.no_plot:
        print('─' * 64)
        print('GENERATING PLOTS')
        print('─' * 64)

        sitl_data   = None
        mujoco_data = None

        if sitl_csv and sitl_csv.exists():
            try:
                sitl_data = _load_csv(sitl_csv)
                plot_timeline(sitl_data,
                              out_dir / f'e2e_timeline_sitl_motor{fault_motor}.png',
                              fault_motor)
            except Exception as exc:
                print(f'[Plot] SITL timeline failed: {exc}')

        if mujoco_csv and mujoco_csv.exists():
            try:
                mujoco_data = _load_csv(mujoco_csv)
                plot_timeline(mujoco_data,
                              out_dir / f'e2e_timeline_mujoco_motor{fault_motor}.png',
                              fault_motor)
            except Exception as exc:
                print(f'[Plot] MuJoCo timeline failed: {exc}')

        if sitl_data and mujoco_data:
            try:
                plot_comparison(sitl_data, mujoco_data,
                                out_dir / f'e2e_comparison_motor{fault_motor}.png',
                                fault_motor)
            except Exception as exc:
                print(f'[Plot] Comparison failed: {exc}')

    # ── summary ───────────────────────────────────────────────────────────
    print('=' * 64)
    print('SUMMARY')
    print('=' * 64)
    if sitl_csv:
        try:
            _print_summary('SITL', _load_csv(sitl_csv), fault_motor)
        except Exception as exc:
            print(f'  [SITL] Could not summarise: {exc}')
    if mujoco_csv:
        try:
            _print_summary('MuJoCo', _load_csv(mujoco_csv), fault_motor)
        except Exception as exc:
            print(f'  [MuJoCo] Could not summarise: {exc}')

    print(f'\nOutput files in: {out_dir}')
    for p in sorted(out_dir.glob('e2e_*')):
        kb = p.stat().st_size // 1024
        print(f'  {p.name:45s}  {kb:4d} kB')

    return 0


if __name__ == '__main__':
    sys.exit(main())
