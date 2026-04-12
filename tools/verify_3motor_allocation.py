#!/usr/bin/env python3
"""
Verify the 3-motor closed-form allocation formulas.

For each failed motor case, this script:
1. Computes motor values from the closed-form formulas
2. Reconstructs thrust, roll, pitch from those motor values
3. Verifies they match the desired commands
4. Computes the residual yaw
5. Checks feasibility (all working motors >= 0)

Uses the Crazyflie legacy mixer sign convention:
  m1 = T - r + p + Y    (front-left,  CW)
  m2 = T - r - p - Y    (front-right, CCW)
  m3 = T + r - p + Y    (back-right,  CW)
  m4 = T + r + p - Y    (back-left,   CCW)

where r = roll/2, p = pitch/2.
"""

import numpy as np

# Forward model: motors -> wrench (in firmware scaled units)
# u = (1/4) * B * m
B = np.array([
    [ 1,  1,  1,  1],   # T
    [-1, -1,  1,  1],   # r (= roll/2)
    [ 1, -1, -1,  1],   # p (= pitch/2)
    [ 1, -1,  1, -1],   # Y (= yaw)
])

def motors_to_wrench(m):
    return (1/4) * B @ m

def wrench_to_motors(u):
    """Standard 4-motor inverse (B^T since B is orthogonal with B^T B = 4I)"""
    return B.T @ u

def three_motor_allocation(T, r, p, failed_motor):
    """
    Closed-form 3-motor allocation in firmware variables.
    Returns (m1, m2, m3, m4) with failed motor = 0.
    """
    if failed_motor == 1:
        m1 = 0
        m2 = 2 * (T - r)
        m3 = 2 * (r - p)
        m4 = 2 * (T + p)
        yaw_res = -T + r - p
    elif failed_motor == 2:
        m1 = 2 * (T - r)
        m2 = 0
        m3 = 2 * (T - p)
        m4 = 2 * (r + p)
        yaw_res = T - r - p
    elif failed_motor == 3:
        m1 = 2 * (p - r)
        m2 = 2 * (T - p)
        m3 = 0
        m4 = 2 * (T + r)
        yaw_res = p - r - T
    elif failed_motor == 4:
        m1 = 2 * (T + p)
        m2 = -2 * (r + p)
        m3 = 2 * (T + r)
        m4 = 0
        yaw_res = T + r + p
    else:
        raise ValueError(f"Invalid failed_motor: {failed_motor}")

    return np.array([m1, m2, m3, m4]), yaw_res

def check_feasibility(motors, failed_motor):
    """Check all working motors are non-negative."""
    for i in range(4):
        if i + 1 == failed_motor:
            continue
        if motors[i] < -1e-9:
            return False
    return True

# ---- Run verification ----
print("=" * 70)
print("3-MOTOR FAULT-TOLERANT ALLOCATION VERIFICATION")
print("=" * 70)

# Test with hover-like conditions
test_cases = [
    ("Hover",           40000, 0,    0,    0),
    ("Small roll",      40000, 200,  0,    0),
    ("Small pitch",     40000, 0,    200,  0),
    ("Roll + pitch",    40000, 100,  100,  0),
    ("Descent",         20000, 0,    0,    0),
    ("Aggressive roll", 40000, 2000, 0,    0),
]

for name, T, R, P, Y in test_cases:
    r = R / 2.0
    p = P / 2.0
    print(f"\n{'─' * 70}")
    print(f"Test: {name} | T={T}, R={R}, P={P} (r={r}, p={p})")
    print(f"{'─' * 70}")

    for fm in range(1, 5):
        motors, yaw_res = three_motor_allocation(T, r, p, fm)
        reconstructed = motors_to_wrench(motors)

        T_err = abs(reconstructed[0] - T)
        r_err = abs(reconstructed[1] - r)
        p_err = abs(reconstructed[2] - p)
        feasible = check_feasibility(motors, fm)

        status = "PASS" if (T_err < 0.01 and r_err < 0.01 and p_err < 0.01) else "FAIL"
        feas_str = "feasible" if feasible else "INFEASIBLE"

        print(f"  Motor {fm} failed: m=[{motors[0]:8.0f}, {motors[1]:8.0f}, {motors[2]:8.0f}, {motors[3]:8.0f}]"
              f"  T_err={T_err:.2f} r_err={r_err:.2f} p_err={p_err:.2f}"
              f"  yaw_res={yaw_res:8.0f}  [{status}] [{feas_str}]")

print(f"\n{'=' * 70}")
print("Verification complete.")
print("=" * 70)
