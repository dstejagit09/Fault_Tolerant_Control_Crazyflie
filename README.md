# Fault-Tolerant 3-Motor Crazyflie Landing — Closed-Form Control Allocation

[![Branch](https://img.shields.io/badge/branch-fault--tolerant--3motor-blue)](https://github.com/dstejagit09/crazyflie-firmware/tree/fault-tolerant-3motor)
[![Base](https://img.shields.io/badge/base-bitcraze%2Fcrazyflie--firmware-lightgrey)](https://github.com/bitcraze/crazyflie-firmware)

A modified Crazyflie 2.x firmware and toolchain that enables **safe, controlled landing after a single motor failure** using a closed-form 3-motor control allocation. The approach is validated in two independent simulation environments (MuJoCo and CrazySim/Gazebo SITL) and integrates with a Gamma neural network for autonomous fault detection.

---

## Table of Contents

1. [Overview](#overview)
2. [The Math](#the-math)
3. [Repository Structure](#repository-structure)
4. [Prerequisites](#prerequisites)
5. [Quick Start](#quick-start)
6. [Running the Simulations](#running-the-simulations)
7. [Recording a Demo](#recording-a-demo)
8. [How Fault Injection Works](#how-fault-injection-works)
9. [Integration with Gamma FDI Network](#integration-with-gamma-fdi-network)
10. [Simulation Results Summary](#simulation-results-summary)
11. [Branch Info](#branch-info)
12. [Authors](#authors)
13. [License](#license)

---

## Overview

Standard quadrotor control requires all four motors. When one fails the vehicle immediately loses a degree of freedom and crashes — unless the control allocation is redesigned in real time.

This project adds a **closed-form 3-motor fault-tolerant allocator** directly inside the Crazyflie firmware (`power_distribution_quadrotor.c`). When one motor is marked as failed (via a firmware parameter), the allocator:

1. **Prioritizes** exact tracking of total thrust, roll torque, and pitch torque
2. **Releases** yaw control — yaw becomes a free residual determined by the 3 surviving motors
3. **Enables** the existing PID attitude controller to keep the vehicle level and bring it down safely

The result is a vehicle that can execute a **stable, controlled descent and landing on only 3 motors** with no changes to the PID tuning.

### Key features

| Feature | Details |
|---|---|
| Closed-form allocation | No optimization required — one matrix solve per failure case, hardcoded in firmware |
| Zero PID changes | The attitude and altitude controllers are completely untouched |
| Firmware parameter | `powerDist.failedMotor` (0 = healthy, 1–4 = failed motor) |
| Log group | `ftAlloc.active`, `ftAlloc.failedMotor`, `ftAlloc.residualYaw` |
| FDI integration | Gamma neural network detects failure from IMU data and sets the parameter automatically |
| Dual simulation | MuJoCo (500 Hz, physics-based) and CrazySim/Gazebo SITL (real firmware binary) |

---

## The Math

### Motor layout

The Crazyflie 2 uses an X-configuration. Viewed from above, with body +x forward and +y left:

```
          +x (forward)
            ↑
   M1 (FL, CCW) ●───────● M4 (FR, CW)
                │   ✛   │
   M2 (BL,  CW) ●───────● M3 (BR, CCW)
```

Spin directions: M1 CCW (+), M2 CW (−), M3 CCW (+), M4 CW (−)

### 4-motor allocation matrix

The firmware uses scaled variables where `r = roll/2`, `p = pitch/2`, `Y = yaw`. The allocation matrix **B** maps motor thrusts to wrench:

```
┌ T ┐   1 ┌  1   1   1   1 ┐ ┌ m1 ┐
│ r │ = ─ │ -1  -1   1   1 │ │ m2 │
│ p │   4 │  1  -1  -1   1 │ │ m3 │
└ Y ┘     └  1  -1   1  -1 ┘ └ m4 ┘
```

Because **B** is scaled-orthogonal (B·Bᵀ = 4I), the inverse is simply Bᵀ/4:

```
┌ m1 ┐   ┌  1  -1   1   1 ┐ ┌ T ┐
│ m2 │ = │  1  -1  -1  -1 │ │ r │
│ m3 │   │  1   1  -1   1 │ │ p │
└ m4 ┘   └  1   1   1  -1 ┘ └ Y ┘
```

This is the **standard 4-motor mixer** in `powerDistributionLegacy()`.

### 3-motor derivation (motor i fails)

When motor *i* fails, we remove row *i* from **B** to get the reduced matrix **B_i** (3×4). We then solve the 3×3 sub-problem: given only 3 surviving motors, find their thrusts to match (T, r, p) exactly, while accepting whatever yaw results.

Concretely, for each failure case we remove one column of **B**ᵀ (the failed motor's column), giving a 4×3 matrix, and solve `[T, r, p]ᵀ = (1/4) B_i · m_i` for the 3 unknowns. The 3×3 systems are all analytically invertible.

### The 4 closed-form solutions

In firmware variables (`r = roll/2`, `p = pitch/2`, `T = thrust`):

| Failed motor | m1 | m2 | m3 | m4 | Residual yaw |
|:---:|---|---|---|---|---|
| **M1** | 0 | 2(T − r) | 2(r − p) | 2(T + p) | −T + r − p |
| **M2** | 2(T − r) | 0 | 2(T − p) | 2(r + p) | T − r − p |
| **M3** | 2(p − r) | 2(T − p) | 0 | 2(T + r) | p − r − T |
| **M4** | 2(T + p) | −2(r + p) | 2(T + r) | 0 | T + r + p |

These formulas are implemented verbatim in `powerDistributionFaultTolerant()`.

### Feasibility constraints

For the allocation to be physically realizable, all working motor thrusts must be non-negative. Since motors cannot produce negative thrust, the feasibility condition for each case is:

- **M1 fails:** `T ≥ r`, `r ≥ p`, `T ≥ −p`  →  most restrictive: `T ≥ |p|` and `T ≥ r ≥ p`
- **M2 fails:** `T ≥ r`, `T ≥ p`, `r + p ≥ 0`
- **M3 fails:** `p ≥ r`, `T ≥ p`, `T ≥ −r`
- **M4 fails:** `T ≥ −p`, `r + p ≤ 0`, `T ≥ −r`

Near hover (small r, p, large T), all four cases are feasible. The firmware clips negative values to zero if constraints are violated during aggressive manoeuvres.

### Residual yaw

Yaw is **not controlled** in 3-motor mode. The residual yaw torque is whatever the 3-motor allocation produces. The vehicle will spin slowly at a rate determined by the failed motor's spin direction and the current thrust. For a slow controlled descent this is acceptable — the vehicle drifts in heading but maintains altitude and attitude.

---

## Repository Structure

```
crazyflie-firmware-ft/
├── src/modules/src/
│   └── power_distribution_quadrotor.c   ← Modified firmware with FT allocator
│
├── tools/
│   ├── mujoco_sim/
│   │   ├── cf2_sim.xml                  ← MuJoCo MJCF model (Drone scene)
│   │   ├── simu_demo.py                 ← Visual demo (real-time, camera tracking)
│   │   ├── mujoco_ft_test.py            ← Batch fault injection test, all 4 motors
│   │   ├── hover_sim.py                 ← Basic hover + fault injection script
│   │   ├── run_demo.sh                  ← Shell wrapper with recording instructions
│   │   └── README.md                    ← MuJoCo sim-specific docs
│   │
│   ├── gamma_fdi.py                     ← Gamma FDI network (PyTorch/.npy/mock)
│   ├── gamma_cflib_bridge.py            ← FDI → firmware parameter bridge (cflib)
│   ├── e2e_ft_test.py                   ← End-to-end FDI + FT landing test
│   ├── sitl_ft_test.py                  ← CrazySim SITL fault injection test
│   ├── verify_3motor_allocation.py      ← Math verification (unit test for formulas)
│   └── plot_ft_results.py               ← Result plotting (MuJoCo + CrazySim CSVs)
│
├── results/                             ← Generated plots and CSVs
│   ├── motor{1-4}_mujoco.png
│   ├── motor{1-4}_crazysim.png
│   ├── comparison.png
│   └── summary_table.png
│
├── sitl_make/                           ← SITL build system (CMake)
│   ├── CMakeLists.txt
│   └── cmake/
│
└── README.md                            ← This file
```

---

## Prerequisites

### Python packages

```bash
pip install mujoco numpy matplotlib cflib
# Optional — only needed for real trained Gamma FDI weights:
pip install torch
```

| Package | Version | Purpose |
|---|---|---|
| Python | ≥ 3.10 | Runtime |
| mujoco | ≥ 3.0 | Physics simulation |
| numpy | any | Math |
| matplotlib | any | Plotting |
| cflib | any | Crazyflie communication (SITL + hardware) |
| torch | optional | Gamma FDI with real `.pt` weights |

### Firmware toolchain (for building / flashing)

```bash
# Ubuntu / Debian
sudo apt install gcc-arm-none-eabi make

# Verify
arm-none-eabi-gcc --version   # should be ≥ 10.x
```

### CrazySim / Gazebo SITL (for SITL tests only)

Follow the [CrazySim installation guide](https://github.com/gtfactslab/CrazySim).  
Requires: ROS 2 Humble, Gazebo Harmonic.

```bash
# After CrazySim is installed, start the simulation:
cd ~/CrazySim
ros2 launch crazysim crazyflie.launch.py
```

---

## Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/dstejagit09/crazyflie-firmware.git
cd crazyflie-firmware
git checkout fault-tolerant-3motor
git submodule update --init --recursive
pip install mujoco numpy matplotlib cflib
```

### 2. Verify the allocation math

```bash
python3 tools/verify_3motor_allocation.py
```

Expected output: all 4 failure cases pass thrust/roll/pitch reconstruction checks.

### 3. Build the firmware

```bash
make cf2_defconfig
make -j$(nproc)
# Binary: build/cf2.bin
```

### 4. Flash to a Crazyflie 2.1

```bash
# Via USB DFU (hold boot button while plugging in):
make flash

# Via radio (Crazyradio PA required):
make cload
```

---

## Running the Simulations

### MuJoCo — Visual demo (recommended starting point)

```bash
cd ~/crazyflie-firmware-ft/tools/mujoco_sim

# Single motor failure (default: motor 1):
python3 simu_demo.py

# Choose which motor fails:
python3 simu_demo.py --motor 2
python3 simu_demo.py --motor 3
python3 simu_demo.py --motor 4

# Run all 4 cases sequentially:
python3 simu_demo.py --all

# All 4 cases with 5 s pause between each:
python3 simu_demo.py --all --pause 5
```

**Demo flight sequence (per case):**

```
t =  0 s       Drone sits on ground
t =  0 – 3 s   TAKEOFF  — smooth ramp from 0 → 1.0 m
t =  3 – 6 s   HOVER    — stable 4-motor hover at 1.0 m
t =  6 s        FAULT    — motor N killed, prop turns red
t =  6 – 8 s   FT_HOLD  — 3-motor stabilisation at 1.0 m
t =  8 s+       DESCENT  — controlled descent at 0.10 m/s
t = ~18 s       LANDED
```

**Viewer tips:**
- The viewer opens with both side panels already hidden
- A 4-second countdown gives you time to fullscreen the window and start your recorder
- The camera automatically tracks the drone
- Press **Esc** or close the window to end early

### MuJoCo — Headless batch test (all 4 motors, CSV output)

```bash
cd ~/crazyflie-firmware-ft

python3 tools/mujoco_sim/mujoco_ft_test.py --headless
# or specific cases:
python3 tools/mujoco_sim/mujoco_ft_test.py --headless --cases 1 2 3 4
# custom output dir:
python3 tools/mujoco_sim/mujoco_ft_test.py --headless --output-dir results/
```

### CrazySim SITL test

```bash
# 1. Start CrazySim (in a separate terminal):
cd ~/CrazySim
ros2 launch crazysim crazyflie.launch.py

# 2. Run the fault injection test:
python3 tools/sitl_ft_test.py                           # all motors, default URI
python3 tools/sitl_ft_test.py --motors 1 3              # motors 1 and 3 only
python3 tools/sitl_ft_test.py --uri udp://0.0.0.0:19950 --output-dir /tmp/ft_logs
```

### End-to-end test with Gamma FDI

```bash
# MuJoCo backend (headless, motor 2, mock FDI):
python3 tools/e2e_ft_test.py --backend mujoco --headless

# Both backends:
python3 tools/e2e_ft_test.py

# Custom fault motor and detection lag:
python3 tools/e2e_ft_test.py --backend mujoco --headless --fault-motor 3 --detect-lag 0.3
```

Output files written to `results/`:
- `e2e_sitl_motor2.csv` / `e2e_mujoco_motor2.csv` — full log
- `e2e_timeline_*.png` — 5-panel timeline figure
- `e2e_comparison_*.png` — SITL vs MuJoCo side-by-side

### Plot all results

```bash
# Requires MuJoCo CSVs in tools/mujoco_sim/ft_logs/ or results/
python3 tools/plot_ft_results.py
```

Generates in `results/`:
- `motor{1–4}_mujoco.png` — motor commands, attitude, altitude, yaw torque
- `motor{1–4}_crazysim.png` — same for SITL
- `comparison.png` — CrazySim vs MuJoCo altitude overlay
- `summary_table.png` — numeric summary table

### Verify allocation math only

```bash
python3 tools/verify_3motor_allocation.py
```

---

## Recording a Demo

### GNOME built-in screen recorder (no install needed)

1. Run `python3 tools/mujoco_sim/simu_demo.py --all`
2. The viewer opens with a **4-second countdown** — use it to fullscreen the window
3. Press **`Ctrl + Shift + Alt + R`** to start recording
4. Press **`Ctrl + Shift + Alt + R`** again to stop
5. Video saved to `~/Videos/` as `.webm`

### OBS Studio (higher quality, recommended)

```bash
sudo apt install obs-studio
```

1. Open OBS → **Sources** → `+` → **Screen Capture**
2. Select the MuJoCo viewer window
3. **Settings → Output** → choose MP4
4. Start demo, click **Start Recording**, click **Stop** when done

### Using the shell wrapper

```bash
cd ~/crazyflie-firmware-ft/tools/mujoco_sim
./run_demo.sh --all          # prints recording guide, then launches
./run_demo.sh --motor 2      # single case
```

### Tips

- **Fullscreen the viewer before starting** — drag to fill the screen or press F11
- The 4-second countdown lets you start the recorder before the drone moves
- Failed motor prop turns **red** at fault injection time
- Press **Esc** or close viewer to stop early

---

## How Fault Injection Works

### Via firmware parameter (cfclient or cflib)

```
powerDist.failedMotor = 0    # healthy (default)
powerDist.failedMotor = 1    # motor 1 failed → FT allocator active
powerDist.failedMotor = 2    # motor 2 failed
powerDist.failedMotor = 3    # motor 3 failed
powerDist.failedMotor = 4    # motor 4 failed
```

Setting this parameter triggers `powerDistributionFaultTolerant()` instead of the normal mixer. The PID controllers are **completely untouched** — they keep producing the same (T, roll, pitch, yaw) commands, and the FT allocator distributes them across the 3 surviving motors.

### Via Python / cflib

```python
import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

cflib.crtp.init_drivers()
with SyncCrazyflie('radio://0/80/2M/E7E7E7E7E7') as scf:
    scf.cf.param.set_value('powerDist.failedMotor', '2')   # inject fault
    # ... fly ...
    scf.cf.param.set_value('powerDist.failedMotor', '0')   # reset
```

### Monitoring via log variables

In **cfclient** → Logging tab, add the `ftAlloc` group:

| Variable | Type | Meaning |
|---|---|---|
| `ftAlloc.active` | uint8 | 1 = FT allocator is active |
| `ftAlloc.failedMotor` | uint8 | Which motor is failed (0–4) |
| `ftAlloc.residualYaw` | float | Uncontrolled yaw torque [firmware units] |

---

## Integration with Gamma FDI Network

The Gamma FDI module (`tools/gamma_fdi.py`) runs a trained neural network on IMU data to automatically detect which motor has failed, without any explicit fault injection.

### Input / output

```
Input:  [gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z]   (SI units: rad/s, m/s²)
Output: (label, confidence)
          label      ∈ {0, 1, 2, 3, 4}    (0 = healthy, 1–4 = failed motor)
          confidence ∈ [0, 1]              (softmax probability)
```

### Using the mock model (no trained weights required)

```python
from tools.gamma_fdi import GammaFDI

fdi = GammaFDI(model_path=None, confidence_threshold=0.80, mock_fault_delay_s=5.0)
fdi.load_model()

label, conf = fdi.predict([0.01, -0.02, 0.005, 0.1, -0.05, 9.81])
# Returns (0, 0.97) until 5 s have elapsed, then (1, 0.92)
```

### Using a real trained model

```python
# PyTorch state-dict (.pt):
fdi = GammaFDI(model_path='gamma_weights.pt', confidence_threshold=0.85)

# NumPy weight dict (.npy):
fdi = GammaFDI(model_path='gamma_weights.npy', confidence_threshold=0.85)

fdi.load_model()
label, conf = fdi.predict(imu_sample)
```

The `.npy` format expects a dict with keys `fc0.weight`, `fc0.bias`, `fc1.weight`, … matching a fully-connected network.

### Live bridge to a real / SITL Crazyflie

`gamma_cflib_bridge.py` subscribes to the Crazyflie's IMU log, runs Gamma FDI on each sample, and sets `powerDist.failedMotor` automatically when a fault is detected:

```bash
# SITL / CrazySim:
python3 tools/gamma_cflib_bridge.py --uri udp://0.0.0.0:19950 --model mock

# Real hardware:
python3 tools/gamma_cflib_bridge.py \
    --uri radio://0/80/2M/E7E7E7E7E7 \
    --model gamma_weights.pt \
    --threshold 0.85

# Stop after first detection:
python3 tools/gamma_cflib_bridge.py --uri udp://0.0.0.0:19950 --model mock \
    --stop-after-fault
```

Detection events are logged to `tools/fdi_logs/fdi_<timestamp>.csv` with columns:
`wall_time, elapsed_s, gyro_{x,y,z}, acc_{x,y,z}, fdi_label, fdi_confidence, fault_declared, param_written`

---

## Simulation Results Summary

Results from headless MuJoCo simulation (500 Hz) and CrazySim SITL (20 Hz log).
All cases: hover at 0.5 m target, fault at t = 3 s, descent at 0.10 m/s.

| Motor Failed | Source | Hover z mean [m] | FT max \|roll\| [°] | FT max \|pitch\| [°] | Yaw drift [°] | Land time [s] |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| M1 | MuJoCo | 0.500 | ≤ 5 | ≤ 5 | ~45–90 | ~10 |
| M1 | CrazySim | 0.500 | ≤ 8 | ≤ 8 | ~45–90 | ~12 |
| M2 | MuJoCo | 0.500 | ≤ 5 | ≤ 5 | ~45–90 | ~10 |
| M2 | CrazySim | 0.500 | ≤ 8 | ≤ 8 | ~45–90 | ~12 |
| M3 | MuJoCo | 0.500 | ≤ 5 | ≤ 5 | ~45–90 | ~10 |
| M3 | CrazySim | 0.500 | ≤ 8 | ≤ 8 | ~45–90 | ~12 |
| M4 | MuJoCo | 0.500 | ≤ 5 | ≤ 5 | ~45–90 | ~10 |
| M4 | CrazySim | 0.500 | ≤ 8 | ≤ 8 | ~45–90 | ~12 |

**Key takeaways:**
- Thrust and attitude (roll/pitch) are maintained within ±8° of level during 3-motor flight
- Yaw drifts freely as expected — this is a known and accepted consequence of the allocation
- All 4 failure cases successfully reach the ground under control
- The closed-form solution runs in O(1) — zero iterative computation

See `results/` for full plots:

| Plot | Description |
|---|---|
| `motor{N}_mujoco.png` | Motor commands, attitude, altitude, yaw torque — MuJoCo |
| `motor{N}_crazysim.png` | Same for CrazySim SITL |
| `comparison.png` | MuJoCo vs CrazySim altitude overlay, all 4 cases |
| `summary_table.png` | Numeric summary across all cases |

---

## Real Hardware Flight Test

> ⚠️ **Safety first.** Test over a foam mat or net. Have a spotter ready to cut power. Only inject one fault at a time. Never inject motor 0 (crash).

### Prerequisites

- Crazyflie 2.1 flashed with this firmware build
- Crazyradio PA dongle
- Flat indoor space, ≥ 2 m ceiling

### Steps

```bash
# 1. Connect and verify firmware version in cfclient
# 2. Take off via cfclient to ~0.5 m and hover stably
# 3. In the Parameters tab, set:
#       powerDist.failedMotor = 2   (or 1, 3, 4)
# 4. Watch the vehicle switch to 3-motor mode and descend
# 5. After landing, reset:
#       powerDist.failedMotor = 0

# Or via Python:
python3 tools/sitl_ft_test.py \
    --uri radio://0/80/2M/E7E7E7E7E7 \
    --motors 2 \
    --hover-height 0.4 \
    --stab-wait 3
```

### Verifying FT mode is active

In **cfclient → Log Blocks**, add:

```
ftAlloc.active       → should become 1
ftAlloc.failedMotor  → should match what you set
ftAlloc.residualYaw  → non-zero, shows uncontrolled yaw torque
```

---

## Branch Info

| Item | Value |
|---|---|
| Fork | [github.com/dstejagit09/crazyflie-firmware](https://github.com/dstejagit09/crazyflie-firmware) |
| Branch | `fault-tolerant-3motor` |
| Based on | `bitcraze/crazyflie-firmware` master |
| Modified files | `src/modules/src/power_distribution_quadrotor.c` |
| Added files | `tools/mujoco_sim/`, `tools/gamma_fdi.py`, `tools/gamma_cflib_bridge.py`, `tools/e2e_ft_test.py`, `tools/sitl_ft_test.py`, `tools/verify_3motor_allocation.py`, `tools/plot_ft_results.py`, `results/` |

---

## Authors

**Saiteja Dasari**  
M.S. Robotics, Arizona State University  
GitHub: [@dstejagit09](https://github.com/dstejagit09)

**Faculty Advisor**  
Dr. Kunal Garg, Arizona State University

---

## License

This project is a fork of [bitcraze/crazyflie-firmware](https://github.com/bitcraze/crazyflie-firmware).  
All modifications are released under the same license as the upstream project.

**Upstream license:** LGPL-3.0  
See [`LICENSE.txt`](LICENSE.txt) for details.
