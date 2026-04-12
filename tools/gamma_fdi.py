#!/usr/bin/env python3
"""
gamma_fdi.py — Gamma Neural Network Fault Detection and Isolation (FDI)
========================================================================
Loads a trained Gamma network (PyTorch .pt or NumPy .npy weights) and
classifies Crazyflie IMU samples into one of five classes:

    0 = healthy (all motors nominal)
    1 = motor 1 failed
    2 = motor 2 failed
    3 = motor 3 failed
    4 = motor 4 failed

Input vector:  [gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z]  (6 floats)
               gyro in rad/s,  acc in m/s²  (standard SI units)

Usage
-----
from gamma_fdi import GammaFDI

fdi = GammaFDI(model_path='gamma_weights.pt', confidence_threshold=0.8)
fdi.load_model()

label, confidence = fdi.predict([0.01, -0.02, 0.005, 0.1, -0.05, 9.81])
if label != 0 and confidence > fdi.confidence_threshold:
    print(f'Motor {label} failure detected  (conf={confidence:.3f})')
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np

# ── typing aliases ─────────────────────────────────────────────────────────────
IMUSample = Sequence[float]   # length-6:  gyro_xyz, acc_xyz

# ── constants ──────────────────────────────────────────────────────────────────
NUM_CLASSES      = 5          # 0 (healthy) + 4 motor-failure classes
INPUT_DIM        = 6          # [gx, gy, gz, ax, ay, az]
DEFAULT_THRESHOLD = 0.80      # confidence threshold for declaring a fault


# ══════════════════════════════════════════════════════════════════════════════
# Internal helper: pure-NumPy MLP forward pass
# ══════════════════════════════════════════════════════════════════════════════

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


class _NumpyMLP:
    """Minimal MLP that runs forward-pass with NumPy weight arrays.

    Expected weight dict keys (matching the PyTorch export convention):
        'fc0.weight'  shape (H0, 6)
        'fc0.bias'    shape (H0,)
        'fc1.weight'  shape (H1, H0)
        'fc1.bias'    shape (H1,)
        ...
        'fcN.weight'  shape (5, H_{N-1})
        'fcN.bias'    shape (5,)
    """

    def __init__(self, weights: dict):
        # collect layers in order
        self._layers: list[tuple[np.ndarray, np.ndarray]] = []
        i = 0
        while f'fc{i}.weight' in weights:
            W = np.asarray(weights[f'fc{i}.weight'], dtype=np.float32)
            b = np.asarray(weights[f'fc{i}.bias'],   dtype=np.float32)
            self._layers.append((W, b))
            i += 1
        if not self._layers:
            raise ValueError(
                "Weight dict contains no 'fc0.weight' / 'fc0.bias' keys. "
                "Expected keys like fc0.weight, fc0.bias, fc1.weight, …"
            )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return softmax probability vector of shape (5,)."""
        h = x.astype(np.float32)
        for k, (W, b) in enumerate(self._layers):
            h = W @ h + b
            if k < len(self._layers) - 1:
                h = _relu(h)
        return _softmax(h)


# ══════════════════════════════════════════════════════════════════════════════
# Mock / dummy model (for testing without real weights)
# ══════════════════════════════════════════════════════════════════════════════

class _MockModel:
    """Dummy model that returns class 0 (healthy) for a configurable delay,
    then randomly triggers one of the four motor-failure classes.

    Parameters
    ----------
    fault_delay_s  : seconds before the first synthetic fault is triggered
    fault_motor    : which motor to report (1-4); None = random each time
    fault_conf     : confidence level reported for fault detections
    healthy_conf   : confidence level reported for healthy detections
    seed           : random seed for reproducibility
    """

    def __init__(
        self,
        fault_delay_s: float = 5.0,
        fault_motor:   Optional[int] = None,
        fault_conf:    float = 0.92,
        healthy_conf:  float = 0.97,
        seed:          int   = 42,
    ):
        self._fault_delay  = fault_delay_s
        self._fault_motor  = fault_motor
        self._fault_conf   = fault_conf
        self._healthy_conf = healthy_conf
        self._rng          = random.Random(seed)
        self._start_time   = time.monotonic()

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return a fake probability vector (5,)."""
        elapsed = time.monotonic() - self._start_time
        probs = np.zeros(NUM_CLASSES, dtype=np.float32)

        if elapsed < self._fault_delay:
            # healthy
            probs[0] = self._healthy_conf
            residual = 1.0 - self._healthy_conf
            for k in range(1, NUM_CLASSES):
                probs[k] = residual / (NUM_CLASSES - 1)
        else:
            # fault
            motor = self._fault_motor if self._fault_motor else self._rng.randint(1, 4)
            probs[motor] = self._fault_conf
            residual = 1.0 - self._fault_conf
            for k in range(NUM_CLASSES):
                if k != motor:
                    probs[k] = residual / (NUM_CLASSES - 1)

        return probs


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

class GammaFDI:
    """Gamma FDI network wrapper.

    Parameters
    ----------
    model_path           : path to .pt (PyTorch state-dict) or .npy (NumPy
                           weight dict) file.  Pass None or 'mock' to use the
                           built-in dummy model (useful for testing).
    confidence_threshold : minimum softmax score to declare a fault (default 0.8)
    mock_fault_delay_s   : seconds before the mock model starts reporting faults
    mock_fault_motor     : which motor the mock model faults (1-4; None=random)
    """

    def __init__(
        self,
        model_path:           Optional[Union[str, Path]] = None,
        confidence_threshold: float = DEFAULT_THRESHOLD,
        mock_fault_delay_s:   float = 5.0,
        mock_fault_motor:     Optional[int] = None,
    ):
        self.model_path           = Path(model_path) if model_path and str(model_path) != 'mock' else None
        self.confidence_threshold = confidence_threshold
        self._mock_fault_delay    = mock_fault_delay_s
        self._mock_fault_motor    = mock_fault_motor
        self._model               = None   # set by load_model()
        self._use_mock            = (model_path is None or str(model_path) == 'mock')

    # ── public interface ───────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Load weights from disk (or initialise the mock model)."""
        if self._use_mock:
            self._model = _MockModel(
                fault_delay_s=self._mock_fault_delay,
                fault_motor=self._mock_fault_motor,
            )
            print('[GammaFDI] Using mock/dummy model (no real weights loaded).')
            return

        path = self.model_path
        if not path.exists():
            raise FileNotFoundError(f'[GammaFDI] Weight file not found: {path}')

        suffix = path.suffix.lower()
        if suffix == '.pt':
            self._model = self._load_pytorch(path)
        elif suffix == '.npy':
            self._model = self._load_numpy(path)
        else:
            raise ValueError(
                f'[GammaFDI] Unsupported weight format: {suffix}. '
                'Provide a .pt (PyTorch state-dict) or .npy (NumPy weight dict) file.'
            )
        print(f'[GammaFDI] Loaded weights from {path}')

    def predict(self, imu_data: IMUSample) -> Tuple[int, float]:
        """Run inference on a single IMU sample.

        Parameters
        ----------
        imu_data : sequence of 6 floats — [gyro_x, gyro_y, gyro_z,
                                             acc_x,  acc_y,  acc_z]

        Returns
        -------
        (label, confidence)
            label      : int in 0-4  (0 = healthy, 1-4 = failed motor)
            confidence : float in [0, 1]  (softmax probability of winning class)
        """
        if self._model is None:
            raise RuntimeError('[GammaFDI] Call load_model() before predict().')

        x = np.asarray(imu_data, dtype=np.float32)
        if x.shape != (INPUT_DIM,):
            raise ValueError(
                f'[GammaFDI] Expected input shape ({INPUT_DIM},), got {x.shape}'
            )

        probs = self._infer(x)          # shape (5,)
        label      = int(np.argmax(probs))
        confidence = float(probs[label])
        return label, confidence

    def is_fault(self, imu_data: IMUSample) -> Tuple[bool, int, float]:
        """Convenience wrapper around predict().

        Returns
        -------
        (fault_detected, motor_label, confidence)
            fault_detected : True if label != 0 AND confidence > threshold
            motor_label    : 0-4
            confidence     : float
        """
        label, confidence = self.predict(imu_data)
        fault_detected = (label != 0) and (confidence >= self.confidence_threshold)
        return fault_detected, label, confidence

    # ── private helpers ────────────────────────────────────────────────────────

    def _infer(self, x: np.ndarray) -> np.ndarray:
        """Dispatch to the right backend."""
        if isinstance(self._model, _MockModel):
            return self._model.predict_proba(x)
        elif isinstance(self._model, _NumpyMLP):
            return self._model.forward(x)
        else:
            # PyTorch model (torch.nn.Module)
            import torch
            with torch.no_grad():
                t = torch.from_numpy(x).unsqueeze(0)   # (1, 6)
                logits = self._model(t)                 # (1, 5)
                probs  = torch.softmax(logits, dim=-1)
                return probs.squeeze(0).numpy()

    @staticmethod
    def _load_pytorch(path: Path):
        """Load a PyTorch state-dict .pt file.

        The file should be either:
          • a saved nn.Module (torch.save(model, path))
          • a state-dict dict saved with torch.save(model.state_dict(), path)
            — in the latter case we reconstruct a _NumpyMLP from the tensors.
        """
        try:
            import torch
        except ImportError:
            raise ImportError(
                '[GammaFDI] PyTorch is not installed. '
                'Install it with "pip install torch" or use a .npy weight file.'
            )

        obj = torch.load(path, map_location='cpu', weights_only=False)

        # Full saved module
        if hasattr(obj, 'forward'):
            obj.eval()
            return obj

        # State-dict: convert tensors → NumPy and wrap in _NumpyMLP
        if isinstance(obj, dict):
            np_weights = {k: v.numpy() for k, v in obj.items()}
            return _NumpyMLP(np_weights)

        raise ValueError(
            f'[GammaFDI] Cannot interpret .pt file contents: {type(obj)}'
        )

    @staticmethod
    def _load_numpy(path: Path) -> _NumpyMLP:
        """Load a NumPy .npy file containing a dict of weight arrays."""
        raw = np.load(path, allow_pickle=True)
        # np.save of a dict is stored as a 0-d object array
        if raw.ndim == 0:
            weights = raw.item()
        elif isinstance(raw, dict):
            weights = raw
        else:
            raise ValueError(
                '[GammaFDI] .npy file must contain a dict of weight arrays '
                '(saved with numpy.save(path, weights_dict, allow_pickle=True)).'
            )
        return _NumpyMLP(weights)


# ── CLI smoke-test ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='GammaFDI smoke-test')
    parser.add_argument('--model',     default=None,
                        help='Path to .pt or .npy weights (omit for mock)')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help='Confidence threshold (default 0.8)')
    parser.add_argument('--delay',     type=float, default=3.0,
                        help='Mock fault delay in seconds (default 3.0)')
    parser.add_argument('--motor',     type=int,   default=None,
                        help='Mock failed motor 1-4 (default random)')
    parser.add_argument('--samples',   type=int,   default=20,
                        help='Number of samples to run (default 20)')
    parser.add_argument('--interval',  type=float, default=0.5,
                        help='Seconds between samples (default 0.5)')
    args = parser.parse_args()

    fdi = GammaFDI(
        model_path=args.model,
        confidence_threshold=args.threshold,
        mock_fault_delay_s=args.delay,
        mock_fault_motor=args.motor,
    )
    fdi.load_model()

    rng = random.Random(0)
    print(f'\n{"t(s)":>6}  {"label":>5}  {"conf":>6}  status')
    print('-' * 35)
    t0 = time.monotonic()
    for _ in range(args.samples):
        # synthetic IMU noise around hover
        imu = [
            rng.gauss(0, 0.05),   # gx
            rng.gauss(0, 0.05),   # gy
            rng.gauss(0, 0.01),   # gz
            rng.gauss(0, 0.5),    # ax
            rng.gauss(0, 0.5),    # ay
            rng.gauss(9.81, 0.1), # az
        ]
        label, conf = fdi.predict(imu)
        status = 'FAULT motor {}'.format(label) if label != 0 and conf >= fdi.confidence_threshold else 'healthy'
        print(f'{time.monotonic()-t0:>6.2f}  {label:>5}  {conf:>6.3f}  {status}')
        time.sleep(args.interval)
