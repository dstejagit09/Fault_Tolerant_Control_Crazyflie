#!/usr/bin/env python3
"""
gamma_cflib_bridge.py — Gamma FDI ↔ Crazyflie cflib bridge
============================================================
Subscribes to live IMU data (gyroscope + accelerometer) from a Crazyflie via
cflib, feeds each sample to the GammaFDI classifier, and — when a fault is
detected with sufficient confidence — writes the firmware parameter
``powerDist.failedMotor`` so that the on-board fault-tolerant allocator
switches to 3-motor mode.

Detection events are appended to a CSV log with timestamp, detected motor, and
confidence score.

Usage
-----
# Hardware Crazyflie (radio):
python3 gamma_cflib_bridge.py --uri radio://0/80/2M/E7E7E7E7E7

# SITL / CrazySim (UDP):
python3 gamma_cflib_bridge.py --uri udp://0.0.0.0:19950 --model mock

# Real weights:
python3 gamma_cflib_bridge.py --uri radio://0/80/2M/E7E7E7E7E7 \\
        --model gamma_weights.pt --threshold 0.85

# Stop after first confirmed fault, land, reset:
python3 gamma_cflib_bridge.py --uri udp://0.0.0.0:19950 --model mock \\
        --stop-after-fault
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# cflib imports — install with:  pip install cflib
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

# local GammaFDI module
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from gamma_fdi import GammaFDI

# ── tuneable constants ─────────────────────────────────────────────────────────

DEFAULT_URI        = 'udp://0.0.0.0:19950'
LOG_PERIOD_MS      = 20          # IMU logging period [ms] → ~50 Hz
DEFAULT_THRESHOLD  = 0.80        # confidence threshold
DEFAULT_OUTPUT_DIR = _HERE / 'fdi_logs'

# Crazyflie log variable names and types
IMU_LOG_VARS = [
    ('gyro.x', 'float'),   # °/s  — cflib delivers in deg/s; we convert to rad/s
    ('gyro.y', 'float'),
    ('gyro.z', 'float'),
    ('acc.x',  'float'),   # g    — cflib delivers in g; we convert to m/s²
    ('acc.y',  'float'),
    ('acc.z',  'float'),
]

DEG2RAD = 3.141592653589793 / 180.0
G_TO_MS2 = 9.80665

# CSV column names
CSV_FIELDNAMES = [
    'wall_time',            # ISO-8601 wall-clock timestamp
    'elapsed_s',            # seconds since bridge started
    'gyro_x_rads',
    'gyro_y_rads',
    'gyro_z_rads',
    'acc_x_ms2',
    'acc_y_ms2',
    'acc_z_ms2',
    'fdi_label',            # 0-4
    'fdi_confidence',
    'fault_declared',       # True/False
    'param_written',        # True if powerDist.failedMotor was set
]


# ══════════════════════════════════════════════════════════════════════════════
# Bridge class
# ══════════════════════════════════════════════════════════════════════════════

class GammaCflibBridge:
    """Connects to a Crazyflie, streams IMU, runs GammaFDI, acts on faults.

    Parameters
    ----------
    uri                : Crazyflie URI string
    fdi                : initialised GammaFDI instance (load_model already called)
    output_dir         : directory for CSV logs
    stop_after_fault   : if True, stop the bridge after the first confirmed fault
    reset_on_exit      : if True, write powerDist.failedMotor=0 on clean shutdown
    """

    def __init__(
        self,
        uri:              str,
        fdi:              GammaFDI,
        output_dir:       Path = DEFAULT_OUTPUT_DIR,
        stop_after_fault: bool = False,
        reset_on_exit:    bool = True,
    ):
        self.uri              = uri
        self.fdi              = fdi
        self.output_dir       = Path(output_dir)
        self.stop_after_fault = stop_after_fault
        self.reset_on_exit    = reset_on_exit

        self._stop_event      = threading.Event()
        self._fault_declared  = False
        self._fault_motor     = 0
        self._start_time      = 0.0
        self._csv_path        = self._make_csv_path()
        self._csv_file        = None
        self._csv_writer      = None
        self._lock            = threading.Lock()   # protects CSV writer + CF param

    # ── public entry point ────────────────────────────────────────────────────

    def run(self) -> None:
        """Connect, stream, detect, act.  Blocks until stopped or fault (if configured)."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self._make_csv_path()

        print(f'[Bridge] Connecting to {self.uri} …')
        cflib.crtp.init_drivers()

        with SyncCrazyflie(self.uri, cf=Crazyflie(rw_cache='./cf_cache')) as scf:
            self._cf = scf.cf
            print('[Bridge] Connected.')
            self._start_time = time.monotonic()

            with open(self._csv_path, 'w', newline='') as csv_file:
                self._csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
                self._csv_writer.writeheader()
                csv_file.flush()

                log_conf = self._build_log_config()
                with scf.cf.log.add_config(log_conf):
                    log_conf.data_received_cb.add_callback(self._imu_callback)
                    log_conf.start()

                    print(
                        f'[Bridge] Streaming IMU at {LOG_PERIOD_MS} ms  '
                        f'threshold={self.fdi.confidence_threshold:.2f}  '
                        f'log={self._csv_path}'
                    )
                    print('[Bridge] Press Ctrl-C to stop.\n')

                    try:
                        while not self._stop_event.is_set():
                            time.sleep(0.1)
                    except KeyboardInterrupt:
                        print('\n[Bridge] Interrupted by user.')
                    finally:
                        log_conf.stop()
                        if self.reset_on_exit and self._fault_declared:
                            self._write_failed_motor(0)
                            print('[Bridge] Reset powerDist.failedMotor → 0')

        print(f'[Bridge] Log saved to {self._csv_path}')

    def stop(self) -> None:
        """Request a clean shutdown."""
        self._stop_event.set()

    # ── private helpers ───────────────────────────────────────────────────────

    def _make_csv_path(self) -> Path:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        return self.output_dir / f'fdi_{ts}.csv'

    @staticmethod
    def _build_log_config() -> LogConfig:
        lc = LogConfig(name='GammaIMU', period_in_ms=LOG_PERIOD_MS)
        for var_name, var_type in IMU_LOG_VARS:
            lc.add_variable(var_name, var_type)
        return lc

    def _imu_callback(self, timestamp: int, data: dict, logconf: LogConfig) -> None:
        """Called from cflib logging thread for every IMU packet."""
        # Convert units: deg/s → rad/s,  g → m/s²
        gx = data['gyro.x'] * DEG2RAD
        gy = data['gyro.y'] * DEG2RAD
        gz = data['gyro.z'] * DEG2RAD
        ax = data['acc.x']  * G_TO_MS2
        ay = data['acc.y']  * G_TO_MS2
        az = data['acc.z']  * G_TO_MS2

        imu_sample = [gx, gy, gz, ax, ay, az]

        # Run FDI
        label, confidence = self.fdi.predict(imu_sample)
        fault_detected = (label != 0) and (confidence >= self.fdi.confidence_threshold)

        elapsed = time.monotonic() - self._start_time
        wall    = datetime.now().isoformat(timespec='milliseconds')

        param_written = False

        if fault_detected and not self._fault_declared:
            with self._lock:
                if not self._fault_declared:
                    self._fault_declared = True
                    self._fault_motor    = label
                    param_written        = True

            # Print detection banner
            print(
                f'[Bridge] *** FAULT DETECTED ***  '
                f'motor={label}  conf={confidence:.3f}  '
                f'elapsed={elapsed:.3f}s'
            )

            # Write firmware parameter
            self._write_failed_motor(label)

            if self.stop_after_fault:
                self._stop_event.set()

        # Log to CSV
        row = {
            'wall_time':       wall,
            'elapsed_s':       f'{elapsed:.4f}',
            'gyro_x_rads':     f'{gx:.6f}',
            'gyro_y_rads':     f'{gy:.6f}',
            'gyro_z_rads':     f'{gz:.6f}',
            'acc_x_ms2':       f'{ax:.6f}',
            'acc_y_ms2':       f'{ay:.6f}',
            'acc_z_ms2':       f'{az:.6f}',
            'fdi_label':       label,
            'fdi_confidence':  f'{confidence:.6f}',
            'fault_declared':  fault_detected,
            'param_written':   param_written,
        }
        with self._lock:
            if self._csv_writer is not None:
                self._csv_writer.writerow(row)

    def _write_failed_motor(self, motor: int) -> None:
        """Set the powerDist.failedMotor firmware parameter."""
        try:
            self._cf.param.set_value('powerDist.failedMotor', str(motor))
            print(f'[Bridge] powerDist.failedMotor ← {motor}')
        except Exception as exc:
            print(f'[Bridge] WARNING: could not set powerDist.failedMotor: {exc}')


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Gamma FDI → Crazyflie cflib bridge',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--uri',        default=DEFAULT_URI,
                   help='Crazyflie URI')
    p.add_argument('--model',      default=None,
                   help='Path to .pt / .npy weights, or "mock" for dummy model')
    p.add_argument('--threshold',  type=float, default=DEFAULT_THRESHOLD,
                   help='Confidence threshold for fault declaration')
    p.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR),
                   help='Directory for CSV detection logs')
    p.add_argument('--stop-after-fault', action='store_true',
                   help='Disconnect after the first confirmed fault')
    p.add_argument('--no-reset',   action='store_true',
                   help='Do NOT reset powerDist.failedMotor to 0 on exit')
    p.add_argument('--mock-delay', type=float, default=5.0,
                   help='Seconds before mock model triggers fault (mock only)')
    p.add_argument('--mock-motor', type=int, default=None, choices=[1,2,3,4],
                   help='Motor to fault in mock mode (default: random)')
    return p


def main() -> None:
    args = _build_parser().parse_args()

    model_path = args.model  # None → mock, 'mock' → mock, else file path

    fdi = GammaFDI(
        model_path=model_path,
        confidence_threshold=args.threshold,
        mock_fault_delay_s=args.mock_delay,
        mock_fault_motor=args.mock_motor,
    )
    fdi.load_model()

    bridge = GammaCflibBridge(
        uri=uri_helper.uri_from_env(fall_back=args.uri),
        fdi=fdi,
        output_dir=Path(args.output_dir),
        stop_after_fault=args.stop_after_fault,
        reset_on_exit=not args.no_reset,
    )

    # clean SIGTERM shutdown
    def _sigterm(sig, frame):
        print('\n[Bridge] SIGTERM — shutting down.')
        bridge.stop()

    signal.signal(signal.SIGTERM, _sigterm)

    bridge.run()


if __name__ == '__main__':
    main()
