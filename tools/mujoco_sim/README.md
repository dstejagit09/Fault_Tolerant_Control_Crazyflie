# MuJoCo Crazyflie Simulation

Minimal Crazyflie 2 MuJoCo environment for fault-tolerance testing and
presentation demos.  No mesh files required — all geometry is schematic.

## Files

| File | Purpose |
|---|---|
| `cf2_sim.xml` | MJCF model — blue/teal Drone scene, CF2 body + sensors |
| `hover_sim.py` | Basic hover + fault-injection script |
| `mujoco_ft_test.py` | Batch fault-injection test, all 4 motors, CSV output |
| `simu_demo.py` | **Presentation demo** — real-time viewer, 4 s countdown, phase transitions |
| `run_demo.sh` | Shell wrapper with recording instructions |

## Quick start

```bash
cd ~/crazyflie-firmware-ft/tools/mujoco_sim

# Install mujoco if not already present
pip install mujoco

# Run a basic hover (no fault):
python3 hover_sim.py

# Fault-inject motor 2, view result:
python3 hover_sim.py --failed-motor 2 --fault-time 3

# Full batch test (headless, all 4 motors):
python3 mujoco_ft_test.py --headless
```

---

## Recording a Demo for Presentation

### 1 · Install a screen recorder (choose one)

**GNOME built-in** (no install, produces `.webm`):
```
# No installation needed — it's built into GNOME Shell
```

**OBS Studio** (recommended — higher quality, `.mp4`):
```bash
sudo apt install obs-studio
# or on newer Ubuntu:
sudo snap install obs-studio
```

**SimpleScreenRecorder** (lightweight alternative):
```bash
sudo apt install simplescreenrecorder
```

---

### 2 · Run the demo

**Single motor failure (motor 1, default):**
```bash
cd ~/crazyflie-firmware-ft/tools/mujoco_sim
python3 simu_demo.py
```

**Specific motor:**
```bash
python3 simu_demo.py --motor 2
python3 simu_demo.py --motor 3
```

**All 4 cases sequentially (2 s pause between each):**
```bash
python3 simu_demo.py --all
```

**Using the shell wrapper (prints recording guide first):**
```bash
./run_demo.sh --motor 1
./run_demo.sh --all
```

---

### 3 · Record with GNOME built-in recorder

1. Start the demo: `python3 simu_demo.py --all`
2. The MuJoCo viewer window opens — **resize it to fill the screen**
3. Press **`Ctrl + Shift + Alt + R`** to start recording
4. Watch the full demo (takeoff → fault → 3-motor recovery → landing)
5. Press **`Ctrl + Shift + Alt + R`** again to stop
6. Video saved to `~/Videos/` as a `.webm` file

---

### 4 · Record with OBS Studio

1. Open OBS Studio
2. In **Sources**, click `+` → **Screen Capture (PipeWire)**  
   or **Window Capture** and select the MuJoCo window
3. In **Settings → Output**, choose MP4 and a save path
4. Start the demo: `python3 simu_demo.py --all`
5. Click **Start Recording** in OBS
6. Click **Stop Recording** when done

---

### 5 · Tips for a clean recording

- **Maximise or resize** the MuJoCo viewer window before starting the recorder
- The viewer title bar shows **"MuJoCo : Drone scene"** — position it prominently
- Phase transitions print in the terminal in real time (`TAKEOFF`, `HOVER`,  
  `FAULT INJECTED`, `DESCENT`, `LANDED`)
- The camera automatically tracks the drone at `distance=2.0 m, elevation=-20°`
- Failed motor prop turns **red** exactly at fault injection time
- Press **Esc** or close the viewer window to end the demo early
- For a shorter clip, use `--motor 2` (single case, ~12 s total)

---

## Demo flight sequence (per case)

```
  t =  0 s   TAKEOFF  — linear ramp 0 → 0.5 m over 2 s
  t =  2 s   HOVER    — stable 4-motor hover for 3 s
  t =  5 s   FAULT    — motor N killed, prop turns red
  t =  5–7 s FT_HOLD  — 3-motor stabilisation at 0.5 m
  t =  7 s+  DESCENT  — controlled descent at 0.10 m/s
  t = ~12 s  LANDED   — stop setpoint
```

## MuJoCo scene visual style

| Element | Description |
|---|---|
| Ground | Blue/teal two-tone checkerboard, ~0.5 m squares, white grid lines |
| Sky | Dark-blue gradient (no bright skybox) |
| Lighting | Soft overhead fill + angled directional light with shadow |
| Drone props | Green = healthy, Red = failed |
| Camera | Behind-right, 20° below horizon, 2 m from drone |
