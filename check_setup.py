"""Check that this laptop is ready: libraries, webcam, face tracking, audio.

    python check_setup.py                 # full check: opens the webcam, plays a short beep
    python check_setup.py --no-camera --no-sound
    python check_setup.py --llm-ping      # also make one real call to the LLM provider

Fix every FAIL before recording data (see README section 22).
"""
import argparse
import importlib
import sys
import time

import config

RESULTS = []


def report(status, name, detail=""):
    RESULTS.append((status, name, detail))
    mark = {"PASS": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]", "INFO": "[INFO]"}[status]
    print(f"{mark} {name}" + (f" - {detail}" if detail else ""))


def check_python():
    v = sys.version_info
    if (3, 10) <= v[:2] <= (3, 12):
        report("PASS", "Python", f"{v.major}.{v.minor}.{v.micro}")
    else:
        report("FAIL", "Python", f"{v.major}.{v.minor} found; use 3.10, 3.11 or 3.12 (MediaPipe 0.10.14 needs it)")


def check_libraries():
    libs = [("numpy", "numpy"), ("cv2", "opencv"), ("mediapipe", "mediapipe"), ("torch", "torch"),
            ("sklearn", "scikit-learn"), ("matplotlib", "matplotlib"), ("sounddevice", "sounddevice"),
            ("soundfile", "soundfile"), ("pyttsx3", "pyttsx3"), ("dotenv", "python-dotenv")]
    ok = True
    for mod, name in libs:
        try:
            m = importlib.import_module(mod)
            report("PASS", name, getattr(m, "__version__", ""))
        except Exception as e:
            ok = False
            report("FAIL", name, f"cannot import ({e.__class__.__name__}: {e}); run pip install -r requirements.txt")
    try:
        import mediapipe as mp
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            report("PASS", "MediaPipe Face Mesh API", "mp.solutions.face_mesh available")
        else:
            report("FAIL", "MediaPipe Face Mesh API", "missing; install mediapipe==0.10.14")
    except Exception:
        pass
    return ok


def check_phrases():
    from src.utils import load_phrases
    try:
        mvp, full = load_phrases(phrase_set="mvp"), load_phrases(phrase_set="full")
        emerg = sum(p["emergency"] for p in full.values())
        report("PASS", "Phrase file", f"{len(mvp)} MVP / {len(full)} full phrases, {emerg} emergency")
    except Exception as e:
        report("FAIL", "Phrase file", str(e))


def check_model_code():
    try:
        import torch
        from src.model import build_model, count_parameters
        model = build_model({"arch": "conv_bigru", "n_classes": 30})
        out = model(torch.zeros(2, config.TARGET_FRAMES, 160))
        assert out.shape == (2, 30)
        report("PASS", "PyTorch model", f"{count_parameters(model):,} parameters, forward pass OK")
    except Exception as e:
        report("FAIL", "PyTorch model", str(e))


def check_camera(index, show):
    """Returns {"cam_fps", "brightness", "face_rate"} or None if the camera failed."""
    import cv2
    import numpy as np
    from src.landmarks import LipTracker, draw_lips
    from src.utils import close_window, draw_text, make_window, open_camera

    cap = open_camera(index)
    if cap is None:
        report("FAIL", "Webcam", f"camera {index if index is not None else config.CAMERA_INDEX} did not open; "
               "try --camera 1, close Zoom/Teams/other camera apps, and allow camera access for desktop apps "
               "in Windows Settings > Privacy > Camera")
        return None
    print("       Look at the camera for 5 seconds...")
    window = "SilentSpeech - camera check"
    if show:
        make_window(window, 640, 480)

    def preview(img, pts=None):
        view = cv2.flip(draw_lips(img.copy(), pts), 1)
        draw_text(view, "Camera check: look at the screen", (14, 30), 0.7, (255, 255, 255), 2)
        cv2.imshow(window, view)
        cv2.waitKey(1)

    raw, light = 0, []
    start = time.perf_counter()
    while time.perf_counter() - start < 2.0:          # camera speed alone (what recording uses)
        ok, frame = cap.read()
        if ok and frame is not None:
            raw += 1
            light.append(float(frame[::8, ::8].mean()))
            if show:
                preview(frame)
    cam_fps = raw / (time.perf_counter() - start)
    brightness = float(np.median(light)) if light else 0.0

    frames = faces = 0
    track_time = 0.0
    start = time.perf_counter()
    with LipTracker() as tracker:
        while time.perf_counter() - start < 3.0:      # with face tracking (what the live app does)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frames += 1
            t0 = time.perf_counter()
            pts = tracker.process(frame)
            if pts is not None:
                faces += 1
                track_time += time.perf_counter() - t0
            if show:
                preview(frame, pts)
    h, w = (frame.shape[:2] if frames else (0, 0))
    cap.release()
    if show:
        close_window(window)
    if frames == 0:
        report("FAIL", "Webcam", "opened but returned no frames (is another app using the camera?)")
        return None
    report("PASS" if cam_fps >= config.MIN_FPS else "WARN", "Webcam",
           f"{w}x{h}, {cam_fps:.1f} FPS" +
           ("" if cam_fps >= config.MIN_FPS else
            f" (below {config.MIN_FPS}, so clips will be rejected: add light in front of you, close camera apps)"))
    dark = brightness < 50
    report("WARN" if dark else "PASS", "Lighting",
           f"picture brightness {brightness:.0f}/255" +
           (" - very dark: turn on a light in front of you, or open the camera's privacy shutter" if dark else ""))
    rate = faces / frames
    if rate >= 0.9:
        report("PASS", "Face detection", f"face found in {rate:.0%} of frames")
    elif rate > 0:
        report("WARN", "Face detection", f"face found in only {rate:.0%} of frames; light your face from the front")
    else:
        report("FAIL", "Face detection", "no face found; sit 40-70 cm away facing the camera, light from the front"
               + (" (the picture is very dark)" if dark else ""))
    if faces:
        ms = track_time / faces * 1000
        report("PASS" if ms < 45 else "WARN", "Face tracking speed",
               f"{ms:.0f} ms per frame" + ("" if ms < 45 else " (slow CPU: close other apps, plug in the charger)"))
    return {"cam_fps": cam_fps, "brightness": brightness, "face_rate": rate}


def check_audio(play):
    try:
        import numpy as np
        import sounddevice as sd
        dev = sd.query_devices(kind="output")
        report("PASS", "Audio output", dev["name"])
        if play:
            sr = 22050
            t = np.arange(int(sr * 0.25)) / sr
            sd.play((0.3 * np.sin(2 * np.pi * 880 * t)).astype(np.float32), sr)
            sd.wait()
            report("INFO", "Audio test", "you should have heard a short beep")
    except Exception as e:
        report("FAIL", "Audio output", f"{e}")
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        report("PASS", "Offline voice (pyttsx3)", f"{len(voices)} system voice(s)")
        engine.stop()
    except Exception as e:
        hint = " (on Linux: sudo apt install espeak-ng)" if sys.platform.startswith("linux") else ""
        report("FAIL", "Offline voice (pyttsx3)", f"{e}{hint}")


def check_project():
    from src.dataset import scan_clips
    clips = scan_clips()
    speakers = sorted({c.speaker for c in clips})
    report("INFO", "Recorded data", f"{len(clips)} clips from {len(speakers)} speaker(s) "
           f"{', '.join(speakers)}" if clips else "no clips yet (python collect_data.py ...)")
    report("INFO", "Trained model", str(config.BASE_MODEL) if config.BASE_MODEL.exists()
           else "none yet (python train.py)")
    audio = config.PHRASE_AUDIO_DIR / config.SPEAKER_ID
    n = len(list(audio.glob("*.wav"))) if audio.exists() else 0
    report("INFO", "Cloned voice", f"{n} phrase clips for {config.SPEAKER_ID}" if n
           else f"none for {config.SPEAKER_ID} yet (notebooks/voice_cloning_colab.ipynb); pyttsx3 will speak")


def check_llm(ping):
    from src.llm_corrector import LLMCorrector
    from src.utils import load_env, load_phrases
    load_env()
    if not config.USE_LLM:
        report("INFO", "LLM layer", "off (USE_LLM = False)")
        return
    llm = LLMCorrector()
    if not llm.available:
        report("WARN", "LLM layer", f"{llm.unavailable_reason}; the app works without it (--no-llm)")
        return
    if not ping:
        report("PASS", "LLM layer", f"{llm.provider} / {llm.model}, key found (use --llm-ping to test a call)")
        return
    phrases = load_phrases(phrase_set="full")
    t0 = time.perf_counter()
    pid, reason, source = llm.choose([("P10", 0.48), ("P28", 0.41), ("P11", 0.06)], ["P09"], phrases)
    ms = (time.perf_counter() - t0) * 1000
    if source == "llm":
        report("PASS", "LLM call", f"{llm.provider} chose {pid} in {ms:.0f} ms: {reason}")
    else:
        report("FAIL", "LLM call", f"no valid answer within {llm.timeout_s}s; check the key, model name and network")
    llm.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--no-window", action="store_true", help="do not show the camera preview")
    ap.add_argument("--no-sound", action="store_true")
    ap.add_argument("--llm-ping", action="store_true")
    args = ap.parse_args()

    check_python()
    libs_ok = check_libraries()
    check_phrases()
    if libs_ok:
        check_model_code()
        if not args.no_camera:
            check_camera(args.camera, show=not args.no_window)
        check_audio(play=not args.no_sound)
        check_project()
    check_llm(args.llm_ping)

    fails = [r for r in RESULTS if r[0] == "FAIL"]
    warns = [r for r in RESULTS if r[0] == "WARN"]
    print(f"\n{len(fails)} failed, {len(warns)} warning(s).")
    print("Ready to record data." if not fails else "Fix the failed checks above (README section 22).")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
