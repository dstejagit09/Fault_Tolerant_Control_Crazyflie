#!/usr/bin/env bash
# run_demo.sh — Launch the Crazyflie fault-tolerant flight demo
# ============================================================
# Runs simu_demo.py with the MuJoCo Drone scene viewer at real-time speed.
# Designed to be called just before starting a screen recording.
#
# Usage:
#   ./run_demo.sh              # default: motor 1 failure
#   ./run_demo.sh --motor 2    # motor 2 failure
#   ./run_demo.sh --all        # all 4 cases sequentially

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_PY="${SCRIPT_DIR}/simu_demo.py"

# ── Recording instructions ──────────────────────────────────────────────────

cat <<'INSTRUCTIONS'
╔══════════════════════════════════════════════════════════════════╗
║       Crazyflie Fault-Tolerant Demo — Recording Guide           ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  GNOME built-in screen recorder (no install needed):            ║
║    Press  Ctrl + Shift + Alt + R  to START recording            ║
║    Press  Ctrl + Shift + Alt + R  again to STOP                 ║
║    Video saved to  ~/Videos/  as  .webm                         ║
║                                                                  ║
║  OBS Studio (higher quality, recommended):                      ║
║    sudo apt install obs-studio                                   ║
║    Open OBS → Add "Screen Capture" source → Start Recording     ║
║                                                                  ║
║  Tips:                                                           ║
║    • Resize/maximise the MuJoCo viewer window BEFORE recording   ║
║    • The viewer title shows "MuJoCo : Drone scene"               ║
║    • Phase transitions print here in the terminal in real time   ║
║    • Press Esc or close the viewer window to stop early          ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

INSTRUCTIONS

# ── Python environment detection ────────────────────────────────────────────
# Try conda/mamba, then pixi, then fall back to plain python3.

PYTHON_CMD="python3"

if command -v conda &>/dev/null && conda env list 2>/dev/null | grep -q crazyflie; then
    echo "[run_demo] Activating conda env 'crazyflie' …"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate crazyflie 2>/dev/null || true
elif command -v pixi &>/dev/null && [ -f "${SCRIPT_DIR}/../../pixi.toml" ]; then
    echo "[run_demo] Using pixi run …"
    PYTHON_CMD="pixi run python"
fi

# ── Verify mujoco is importable ────────────────────────────────────────────

if ! ${PYTHON_CMD} -c "import mujoco" 2>/dev/null; then
    echo ""
    echo "ERROR: 'mujoco' Python package not found."
    echo "Install with:  pip install mujoco"
    echo ""
    exit 1
fi

# ── Launch demo ─────────────────────────────────────────────────────────────

echo "[run_demo] Starting demo … (MuJoCo viewer will open)"
echo "[run_demo] Command: ${PYTHON_CMD} ${DEMO_PY} $*"
echo ""

exec ${PYTHON_CMD} "${DEMO_PY}" "$@"
