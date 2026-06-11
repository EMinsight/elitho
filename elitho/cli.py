"""Command-line interface for elitho."""

import subprocess
import sys
from pathlib import Path


def run_gui():
    """Launch the Streamlit GUI for elitho."""
    gui_path = Path(__file__).parent / "gui.py"

    if not gui_path.exists():
        print(f"Error: GUI file not found at {gui_path}", file=sys.stderr)
        sys.exit(1)

    result = subprocess.run([sys.executable, "-m", "streamlit", "run", str(gui_path)])
    sys.exit(result.returncode)


if __name__ == "__main__":
    run_gui()
