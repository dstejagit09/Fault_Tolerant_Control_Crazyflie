#!/usr/bin/env python3
"""
sitl_ft_test.py — CrazySim SITL fault-injection test script
============================================================
Tests the fault-tolerant 3-motor allocation for each motor (1–4).

For each motor under test:
  1. Connect to the SITL Crazyflie
  2. Take off to 0.5 m and wait 3 s for Kalman filter / attitude to stabilise
  3. Inject fault via  powerDist.failedMotor  parameter
  4. Log at 50 ms:  motor.m1-m4, ftAlloc.active, ftAlloc.residualYaw,
                    stabilizer.roll/pitch/yaw, stateEstimate.z
  5. Command slow descent at 0.1 m/s while logging
  6. Land, save log to CSV, reset fault, disconnect
  7. Repeat for motors 1–4 (or a subset via --motors flag)

Safety kill: if |roll| or |pitch| exceeds 60°, cut all motors immediately.

--- URI options ---
CrazySim SITL (UDP, default):  udp://0.0.0.0:19950
Simulated radio:               radio://0/80/2M/E7E7E7E7E7

--- Firmware log/param notes ---
The  ftAlloc.*  variables live in power_distribution_quadrotor.c and are only
present when the build uses that file (hardware target or a custom SITL build
that includes it).  power_distribution_sitl.c  (the stock SITL file) does NOT
expose them.  This script will warn and skip those variables gracefully if they
are absent from the TOC, so it still works with a plain SITL build.

--- Usage ---
python3 sitl_ft_test.py                         # all motors, default URI
python3 sitl_ft_test.py --motors 1 3            # only motors 1 and 3
python3 sitl_ft_test.py --uri udp://0.0.0.0:19950 --output-dir /tmp/ft_logs
"""

import argparse
import csv
import os
import threading
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils import uri_helper

# ── tuneable constants ────────────────────────────────────────────────────────

DEFAULT_URI      = 'udp://0.0.0.0:19950'
HOVER_HEIGHT_M   = 0.5       # target hover altitude [m]
STAB_WAIT_S      = 3.0       # stabilisation time before fault injection [s]
TAKEOFF_RAMP_S   = 2.0       # duration of altitude ramp-up [s]
DESCENT_RATE_MS  = 0.1       # descent speed [m/s]
LOG_PERIOD_MS    = 50        # logging period [ms]
SAFETY_ANGLE_DEG = 60.0      # roll/pitch kill threshold [°]
LAND_THRESHOLD_M = 0.05      # consider landed below this altitude [m]
INTER_TEST_WAIT_S = 5.0      # pause between consecutive motor tests [s]

DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'sitl_logs'
)

# ── log / param variable names ────────────────────────────────────────────────
# motor.m1-m4: uint16 in power_distribution_sitl.c / uint32 in quadrotor variant
MOTOR_VARS = [
    ('motor.m1', 'uint16_t'),
    ('motor.m2', 'uint16_t'),
    ('motor.m3', 'uint16_t'),
    ('motor.m4', 'uint16_t'),
]
# ftAlloc.* only available when power_distribution_quadrotor.c is linked
FT_ALLOC_VARS = [
    ('ftAlloc.active',      'uint8_t'),
    ('ftAlloc.failedMotor', 'uint8_t'),
    ('ftAlloc.residualYaw', 'float'),
]
ATTITUDE_VARS = [
    ('stabilizer.roll',  'float'),
    ('stabilizer.pitch', 'float'),
    ('stabilizer.yaw',   'float'),
    ('stateEstimate.z',  'float'),
]

CSV_FIELDNAMES = [
    'timestamp_ms',
    'elapsed_after_fault_ms',
    'phase',
    'motor.m1', 'motor.m2', 'motor.m3', 'motor.m4',
    'ftAlloc.active', 'ftAlloc.failedMotor', 'ftAlloc.residualYaw',
    'stabilizer.roll', 'stabilizer.pitch', 'stabilizer.yaw',
    'stateEstimate.z',
]


# ── SafetyMonitor ─────────────────────────────────────────────────────────────

class SafetyMonitor:
    """
    Background thread that monitors attitude and cuts motors immediately if
    |roll| or |pitch| exceeds SAFETY_ANGLE_DEG.
    """

    def __init__(self, cf: Crazyflie, threshold_deg: float = SAFETY_ANGLE_DEG):
        self._cf         = cf
        self._threshold  = threshold_deg
        self._roll       = 0.0
        self._pitch      = 0.0
        self._lock       = threading.Lock()
        self._killed     = False
        self._stop_evt   = threading.Event()

    # called from log callback (different thread)
    def update_attitude(self, roll: float, pitch: float) -> None:
        with self._lock:
            self._roll  = roll
            self._pitch = pitch

    @property
    def killed(self) -> bool:
        return self._killed

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name='SafetyMonitor')
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            with self._lock:
                roll, pitch = self._roll, self._pitch
            if abs(roll) > self._threshold or abs(pitch) > self._threshold:
                print(
                    f'\n[SAFETY KILL] roll={roll:.1f}°  pitch={pitch:.1f}°  '
                    f'exceeds ±{self._threshold:.0f}° — cutting motors!'
                )
                self._cf.commander.send_stop_setpoint()
                self._killed = True
                self._stop_evt.set()
                return
            time.sleep(0.02)


# ── flight helpers ────────────────────────────────────────────────────────────

def wait_for_position_estimator(scf: SyncCrazyflie, timeout_s: float = 30.0) -> None:
    """
    Block until the Kalman filter reports a valid altitude (z > 0.02 m).
    Times out gracefully after `timeout_s` seconds.
    """
    print('  Waiting for position estimator …', end='', flush=True)
    log = LogConfig('_PosInit', period_in_ms=100)
    log.add_variable('stateEstimate.z', 'float')
    t0 = time.time()
    z_final = 0.0
    with SyncLogger(scf, log) as sl:
        for _, _, data in sl:
            z_final = data.get('stateEstimate.z', 0.0)
            if z_final > 0.02 or (time.time() - t0) > timeout_s:
                break
    print(f' ready (z={z_final:.3f} m)')


def ramp_takeoff(cf: Crazyflie, safety: SafetyMonitor,
                 target_z: float = HOVER_HEIGHT_M,
                 ramp_s: float   = TAKEOFF_RAMP_S) -> bool:
    """
    Linearly ramp altitude setpoint from 0 to target_z over ramp_s seconds.
    Returns True on success, False if safety-killed.
    """
    steps    = max(1, int(ramp_s / (LOG_PERIOD_MS / 1000.0)))
    dt       = ramp_s / steps
    for i in range(steps + 1):
        if safety.killed:
            return False
        z = target_z * i / steps
        cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, z)
        time.sleep(dt)
    return True


def hold_hover(cf: Crazyflie, safety: SafetyMonitor,
               z: float, duration_s: float) -> bool:
    """Hold altitude z for duration_s. Returns False if killed."""
    dt   = LOG_PERIOD_MS / 1000.0
    t0   = time.time()
    while time.time() - t0 < duration_s:
        if safety.killed:
            return False
        cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, z)
        time.sleep(dt)
    return True


def controlled_descent(cf: Crazyflie, safety: SafetyMonitor,
                       start_z: float,
                       rate_ms: float  = DESCENT_RATE_MS,
                       land_z: float   = LAND_THRESHOLD_M) -> None:
    """
    Descend from start_z to land_z at rate_ms [m/s].
    Stops early on safety kill.
    """
    dt      = LOG_PERIOD_MS / 1000.0
    current = start_z
    while current > land_z and not safety.killed:
        current = max(land_z, current - rate_ms * dt)
        cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, current)
        time.sleep(dt)


def cut_motors(cf: Crazyflie) -> None:
    """Send stop setpoint and idle a moment."""
    for _ in range(10):
        cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, 0.0)
        time.sleep(0.02)
    cf.commander.send_stop_setpoint()
    time.sleep(0.3)


# ── log config builder ────────────────────────────────────────────────────────

def build_log_config(cf: Crazyflie) -> tuple[LogConfig, set]:
    """
    Build a LogConfig with all desired variables.
    Variables absent from the firmware TOC are silently skipped.
    Returns (LogConfig, set_of_available_variable_names).
    """
    toc_vars   = set(cf.log.toc.toc.keys()) if hasattr(cf.log.toc, 'toc') else set()
    available  = set()
    log_conf   = LogConfig('FaultTest', period_in_ms=LOG_PERIOD_MS)

    all_vars = MOTOR_VARS + FT_ALLOC_VARS + ATTITUDE_VARS
    for name, vtype in all_vars:
        group, var = name.split('.')
        # Check TOC: toc is {group: {var: LogTocElement}}
        try:
            _ = cf.log.toc.toc[group][var]
            log_conf.add_variable(name, vtype)
            available.add(name)
        except (KeyError, AttributeError):
            pass  # variable not in this firmware build — skip silently

    if not available:
        raise RuntimeError('No log variables found in TOC — is the firmware running?')

    missing = {n for n, _ in all_vars} - available
    if missing:
        print(f'  [INFO] Variables not in TOC (skipped): {sorted(missing)}')

    return log_conf, available


# ── single motor test ─────────────────────────────────────────────────────────

def run_motor_test(uri: str, failed_motor: int, output_dir: str) -> str | None:
    """
    Run the full fault-injection test for one motor.
    Returns the path to the saved CSV, or None on fatal error / safety kill.
    """
    print(f'\n{"=" * 62}')
    print(f'  MOTOR {failed_motor} FAULT INJECTION TEST')
    print(f'{"=" * 62}')

    log_rows            = []
    fault_start_ms: list[int | None] = [None]
    phase               = ['init']
    csv_path            = os.path.join(output_dir, f'motor{failed_motor}_fault.csv')

    cflib.crtp.init_drivers(enable_debug_driver=False)

    try:
        with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
            cf = scf.cf

            # ── reset any lingering fault ────────────────────────────────
            print('  Resetting powerDist.failedMotor → 0')
            cf.param.set_value('powerDist.failedMotor', '0')
            time.sleep(0.3)

            # ── wait for Kalman filter initialisation ────────────────────
            wait_for_position_estimator(scf)

            # ── build log config (skip vars absent from TOC) ─────────────
            log_conf, available_vars = build_log_config(cf)

            def log_callback(timestamp_ms, data, _logconf):
                roll  = data.get('stabilizer.roll',  0.0)
                pitch = data.get('stabilizer.pitch', 0.0)
                safety_mon.update_attitude(roll, pitch)

                elapsed = (timestamp_ms - fault_start_ms[0]
                           if fault_start_ms[0] is not None else None)

                row = {
                    'timestamp_ms':           timestamp_ms,
                    'elapsed_after_fault_ms': elapsed,
                    'phase':                  phase[0],
                    # motors (default 0 if not available)
                    'motor.m1':               data.get('motor.m1', 0),
                    'motor.m2':               data.get('motor.m2', 0),
                    'motor.m3':               data.get('motor.m3', 0),
                    'motor.m4':               data.get('motor.m4', 0),
                    # FT allocator state (default 0/0.0 if not in this build)
                    'ftAlloc.active':         data.get('ftAlloc.active',      0),
                    'ftAlloc.failedMotor':    data.get('ftAlloc.failedMotor', 0),
                    'ftAlloc.residualYaw':    data.get('ftAlloc.residualYaw', 0.0),
                    # attitude / position
                    'stabilizer.roll':        roll,
                    'stabilizer.pitch':       pitch,
                    'stabilizer.yaw':         data.get('stabilizer.yaw',   0.0),
                    'stateEstimate.z':        data.get('stateEstimate.z',  0.0),
                }
                log_rows.append(row)

            cf.log.add_config(log_conf)
            log_conf.data_received_cb.add_callback(log_callback)

            # ── safety monitor (starts watching after log_callback is set) ─
            safety_mon = SafetyMonitor(cf)
            safety_mon.start()

            try:
                # ── take off ─────────────────────────────────────────────
                log_conf.start()
                phase[0] = 'takeoff'
                print(f'  Taking off to {HOVER_HEIGHT_M:.1f} m …')
                if not ramp_takeoff(cf, safety_mon):
                    print('  [SAFETY] Killed during takeoff — aborting test.')
                    return None

                # ── stabilisation hover ───────────────────────────────────
                phase[0] = 'stabilise'
                print(f'  Hovering — stabilisation wait {STAB_WAIT_S:.0f} s …')
                if not hold_hover(cf, safety_mon, HOVER_HEIGHT_M, STAB_WAIT_S):
                    print('  [SAFETY] Killed during stabilisation.')
                    return None

                # ── inject fault ──────────────────────────────────────────
                print(f'  Injecting fault: powerDist.failedMotor = {failed_motor}')
                phase[0] = 'fault_active'
                cf.param.set_value('powerDist.failedMotor', str(failed_motor))
                fault_start_ms[0] = int(time.time() * 1000)
                time.sleep(0.1)   # let param propagate

                if safety_mon.killed:
                    print('  [SAFETY] Killed immediately on fault injection.')
                    return None

                # ── slow descent ──────────────────────────────────────────
                print(f'  Descending at {DESCENT_RATE_MS} m/s …')
                controlled_descent(cf, safety_mon, HOVER_HEIGHT_M)

                # ── land ──────────────────────────────────────────────────
                if not safety_mon.killed:
                    phase[0] = 'landing'
                    print('  Landing …')
                    cut_motors(cf)
                else:
                    print('  [SAFETY] Vehicle killed during descent.')

            finally:
                safety_mon.stop()
                log_conf.stop()
                # always clear the fault before disconnecting
                try:
                    cf.param.set_value('powerDist.failedMotor', '0')
                except Exception:
                    pass

    except Exception as exc:
        print(f'  [ERROR] {exc}')
        return None

    # ── save CSV ──────────────────────────────────────────────────────────
    if not log_rows:
        print('  [WARNING] No log rows collected — CSV not saved.')
        return None

    os.makedirs(output_dir, exist_ok=True)
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES,
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(log_rows)

    print(f'  Saved {len(log_rows)} rows → {csv_path}')

    # ── print quick summary ───────────────────────────────────────────────
    fault_rows = [r for r in log_rows if r['phase'] == 'fault_active']
    if fault_rows:
        rolls   = [r['stabilizer.roll']  for r in fault_rows]
        pitches = [r['stabilizer.pitch'] for r in fault_rows]
        zs      = [r['stateEstimate.z']  for r in fault_rows]
        ft_on   = sum(1 for r in fault_rows if r['ftAlloc.active'])
        print(f'  Fault-active summary ({len(fault_rows)} samples):')
        print(f'    roll   min/max: {min(rolls):+.1f}° / {max(rolls):+.1f}°')
        print(f'    pitch  min/max: {min(pitches):+.1f}° / {max(pitches):+.1f}°')
        print(f'    z      min/max: {min(zs):.3f} m / {max(zs):.3f} m')
        print(f'    ftAlloc.active samples = {ft_on}/{len(fault_rows)}')

    return csv_path


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='CrazySim SITL fault-injection test — motors 1–4',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'URI examples:\n'
            '  SITL/UDP (default) : udp://0.0.0.0:19950\n'
            '  Radio sim          : radio://0/80/2M/E7E7E7E7E7\n'
        ),
    )
    parser.add_argument(
        '--uri', default=DEFAULT_URI,
        help='Crazyflie connection URI (default: %(default)s)',
    )
    parser.add_argument(
        '--motors', nargs='+', type=int, choices=[1, 2, 3, 4],
        default=[1, 2, 3, 4],
        help='Motors to test, e.g. --motors 1 3  (default: all)',
    )
    parser.add_argument(
        '--output-dir', default=DEFAULT_OUTPUT_DIR,
        help='Directory for CSV logs (default: %(default)s)',
    )
    parser.add_argument(
        '--hover-height', type=float, default=HOVER_HEIGHT_M,
        help=f'Takeoff altitude in metres (default: {HOVER_HEIGHT_M})',
    )
    parser.add_argument(
        '--stab-wait', type=float, default=STAB_WAIT_S,
        help=f'Stabilisation wait in seconds (default: {STAB_WAIT_S})',
    )
    args = parser.parse_args()

    # allow CF_URI env-var override (cflib convention)
    uri = uri_helper.uri_from_env(default=args.uri)

    print('CrazySim SITL fault-injection test')
    print(f'  URI:        {uri}')
    print(f'  Motors:     {args.motors}')
    print(f'  Output dir: {args.output_dir}')
    print(f'  Hover:      {args.hover_height} m  |  Stab wait: {args.stab_wait} s')
    print(f'  Safety kill at ±{SAFETY_ANGLE_DEG:.0f}°')

    results: dict[int, str | None] = {}
    for i, motor_id in enumerate(args.motors):
        results[motor_id] = run_motor_test(uri, motor_id, args.output_dir)
        if i < len(args.motors) - 1:
            print(f'\nPausing {INTER_TEST_WAIT_S:.0f} s before next test …')
            time.sleep(INTER_TEST_WAIT_S)

    print(f'\n{"=" * 62}')
    print('ALL TESTS COMPLETE')
    print(f'{"=" * 62}')
    for motor_id, path in results.items():
        status = path if path else '[FAILED / SAFETY KILL / NO DATA]'
        print(f'  Motor {motor_id}: {status}')


if __name__ == '__main__':
    main()
