"""All SilentSpeech settings in one place.

Change behaviour here instead of editing code. Secrets (API keys) belong in
`.env`, never in this file.
"""
import os
from pathlib import Path

# Hide MediaPipe/TensorFlow Lite start-up chatter (it looks like errors but isn't).
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parent

# Everything the project writes (data, models, audio, results) lives under
# WORK_DIR. It defaults to the project folder; set SILENTSPEECH_WORKDIR to
# point somewhere else (the synthetic smoke test uses this).
WORK_DIR = Path(os.environ.get("SILENTSPEECH_WORKDIR", ROOT)).resolve()

# --- Camera ----------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH, FRAME_HEIGHT = 640, 480

# --- Clips and features ----------------------------------------------------
RECORD_SECONDS = 2.5          # one push-to-talk / data-collection window
TARGET_FRAMES = 64            # every clip is resampled to this many frames
COUNTDOWN_SECONDS = 3         # 3-2-1 before each recorded clip

# --- Phrases ---------------------------------------------------------------
PHRASE_FILE = ROOT / "phrases" / "phrases_en.json"
PHRASE_SET = "mvp"            # "mvp" = P01-P30, "full" = all 50
LANGUAGE = "en"               # XTTS-v2 language code used for the cloned voice

# --- Decision thresholds (tune on the validation set: evaluate.py --split val)
CONF_ACCEPT = 0.70            # speak directly at or above this confidence
CONF_MIN = 0.40               # below this, ask the user to repeat
EMERGENCY_MIN = 0.40          # any emergency phrase at or above this raises an alert

# --- LLM correction layer --------------------------------------------------
USE_LLM = True
LLM_PROVIDER = "claude"       # claude | gemini | openai | ollama
LLM_TIMEOUT_S = 2.0           # fall back to the top-1 prediction if slower
LLM_CONTEXT_PHRASES = 5       # how many recent phrases are sent as context
# Small, fast models suit a 2-second budget. Providers retire model names over
# time: if a call fails with "model not found", check the provider's model list.
LLM_MODELS = {
    "claude": "claude-haiku-5-5",
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4.1-mini",
    "ollama": "llama3.2:3b",
}
OLLAMA_URL = "http://localhost:11434"
LLM_ENDPOINTS = {
    "claude": "https://api.anthropic.com/v1/messages",
    "openai": "https://api.openai.com/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
}

# --- Voice -----------------------------------------------------------------
VOICE_MODE = "cloned"         # "cloned" (pre-rendered clips) or "default" (pyttsx3)
SPEAKER_ID = "s01"            # whose cloned voice and enrolled model to use
TTS_RATE = 165                # pyttsx3 words per minute
PHRASE_GAP_S = 0.25           # silence between phrases when speaking a sentence

# --- Privacy ---------------------------------------------------------------
SAVE_VIDEO = False            # save raw video too (only for poster images, with consent)

# --- Automatic clip quality checks (collect_data.py, enroll.py) -------------
MAX_FACE_LOST_FRAC = 0.10     # reject if the face is lost in more than 10% of frames
MIN_FPS = 20                  # reject if the camera delivered fewer frames per second
MIN_LIP_MOTION = 0.06         # reject if lip opening/width changed less than this
                              # (in eye-distance units: the lips barely moved).
                              # Real mouthing measured 0.16-0.35; a missed clip 0.04.

# --- Auto mode (start/stop detection in app.py) ------------------------------
# Starting values; watch the live MAR/motion readout in the status bar and tune.
AUTO_MAR_THRESHOLD = 0.12     # mouth aspect ratio that counts as "mouth open"
AUTO_MOTION_THRESHOLD = 0.010 # mean lip movement per frame (eye-distance units)
AUTO_START_FRAMES = 3         # consecutive active frames needed to start a capture
AUTO_STILL_SECONDS = 0.5      # stop after the lips are still this long
AUTO_PREROLL_SECONDS = 0.3    # frames kept from just before the trigger
AUTO_MIN_SECONDS = 0.5        # ignore shorter blips
AUTO_MAX_SECONDS = 4.0        # force a stop after this long
AUTO_COOLDOWN_SECONDS = 0.8   # pause before re-arming after a capture

# --- Training --------------------------------------------------------------
SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 80
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LR_FACTOR = 0.5               # ReduceLROnPlateau
LR_PATIENCE = 5
EARLY_STOP_PATIENCE = 12
LABEL_SMOOTHING = 0.1
VAL_FRACTION = 0.15           # of sessions 1-2, stratified by speaker and phrase
TEST_SESSIONS = (3,)          # held out completely: a new day and new lighting

# Data augmentation (see src/augment.py). Set a value to 0 to switch it off.
AUGMENT = {
    "mirror_prob": 0.5,       # horizontal mirroring
    "scale": 0.10,            # +/-10% mouth size
    "rotate_deg": 8.0,        # +/-8 degrees head tilt
    "shift": 0.02,            # +/-2% translation jitter
    "stretch": 0.2,           # 0.8x-1.2x speaking speed
    "noise_std": 0.01,        # Gaussian landmark jitter
    "frame_dropout": 0.05,    # up to 5% frozen/dropped frames
}

# --- Quick start (start.py) --------------------------------------------------
# Ten useful, visually varied phrases for a first demo (includes the README's
# "I need water" -> "Call my mother" demo and two emergency phrases).
QUICKSTART_PHRASES = ["P01", "P09", "P10", "P12", "P16", "P17", "P22", "P23", "P28", "P30"]
QUICKSTART_REPS = 4           # calibration clips per phrase (instant templates need 3+)
QUICKSTART_COUNTDOWN = 2      # seconds of 3-2-1 before each calibration clip
FOCUS_PHRASES = 6             # F in the live app: use only this many of your most reliable phrases
TRAIN_MORE_REPS = 4           # start.py --train-more: extra clips of every phrase per round
LEARN_FROM_CHOICES = True     # keep live clips the user confirms with 1/2/3 as new examples
LEARN_SESSION = 9             # session number for those learned clips (data/raw/<speaker>/<id>/9_xx.npy)

# --- Enrolment (enroll.py) -------------------------------------------------
ENROLL_SECONDS = 60           # fine-tuning time budget
ENROLL_LR = 1e-3
ENROLL_MAX_EPOCHS = 300

# --- Paths -----------------------------------------------------------------
DATA_DIR = WORK_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
VIDEO_DIR = DATA_DIR / "video"
METADATA_FILE = DATA_DIR / "metadata.csv"
MODEL_DIR = WORK_DIR / "models"
BASE_MODEL = MODEL_DIR / "lipreader.pt"
LABEL_MAP_FILE = MODEL_DIR / "label_map.json"
AUDIO_DIR = WORK_DIR / "audio"
VOICE_SAMPLE_DIR = AUDIO_DIR / "voice_samples"
PHRASE_AUDIO_DIR = AUDIO_DIR / "phrase_audio"
RESULTS_DIR = WORK_DIR / "results"
