"""Put a SilentSpeech icon on the Windows desktop (double-click it to open the live app).

    python tools/make_shortcut.py          (or double-click make_shortcut.bat)

It draws the app's logo as assets/silentspeech.ico and creates "SilentSpeech"
on the desktop, pointing to app.bat. Run it again any time; it just replaces
the shortcut.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import ui  # noqa: E402

ICON = ROOT / "assets" / "silentspeech.ico"


def make_icon(path=ICON, size=256):
    """The header logo as a Windows icon with transparent corners."""
    import cv2
    from PIL import Image

    bgr = np.zeros((size, size, 3), np.uint8)
    ui.logo(bgr, 0, 0, size)
    alpha = np.zeros((size * 4, size * 4), np.uint8)               # smooth corners: mask drawn 4x, scaled down
    ui.rounded_rect(alpha, 0, 0, size * 4 - 1, size * 4 - 1, int(0.26 * size * 4), 255)
    alpha = cv2.resize(alpha, (size, size), interpolation=cv2.INTER_AREA)
    rgba = np.dstack([bgr[:, :, ::-1], alpha])
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(path, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64),
                                                    (128, 128), (256, 256)])
    return path


def desktop_dir():
    """The real desktop folder (it may be inside OneDrive)."""
    out = subprocess.run(["powershell", "-NoProfile", "-Command", "[Environment]::GetFolderPath('Desktop')"],
                         capture_output=True, text=True, timeout=30)
    path = out.stdout.strip()
    return Path(path) if path else Path.home() / "Desktop"


def make_shortcut(name="SilentSpeech", target=ROOT / "app.bat", icon=ICON):
    link = desktop_dir() / f"{name}.lnk"
    script = (
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:SS_LINK); "
        "$s.TargetPath = $env:SS_TARGET; $s.WorkingDirectory = $env:SS_DIR; "
        "$s.IconLocation = $env:SS_ICON; $s.WindowStyle = 7; "
        "$s.Description = 'SilentSpeech: silent lip reading that speaks in your own voice'; $s.Save()")
    env = {**os.environ, "SS_LINK": str(link), "SS_TARGET": str(target), "SS_DIR": str(ROOT), "SS_ICON": str(icon)}
    subprocess.run(["powershell", "-NoProfile", "-Command", script], env=env, check=True, timeout=60)
    return link


def main():
    if sys.platform != "win32":
        sys.exit("Desktop shortcuts are made this way on Windows only; on macOS / Linux run: sh start.sh")
    icon = make_icon()
    link = make_shortcut()
    print(f"Icon:     {icon}")
    print(f"Shortcut: {link}")
    print("Double-click 'SilentSpeech' on your desktop to open the live app.")


if __name__ == "__main__":
    main()
