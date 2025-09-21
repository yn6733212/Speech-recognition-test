import os
import sys
import logging
import datetime
import subprocess
import requests
import shutil
import tarfile
import speech_recognition as sr
import whisper
from flask import Flask, request, jsonify
from pydub import AudioSegment, effects
import warnings

# -------------------- לוגים --------------------
LOG_LEVEL = logging.INFO
def setup_logging():
    fmt = "%(asctime)s | %(message)s"
    datefmt = "%H:%M:%S"

    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)
    for h in list(root.handlers):
        root.removeHandler(h)

    out_handler = logging.StreamHandler(sys.stdout)
    out_handler.setLevel(LOG_LEVEL)
    out_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(out_handler)

    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=ResourceWarning)

setup_logging()
log = logging.getLogger(__name__)

GREEN = "\033[92m"
RESET = "\033[0m"

def glog(msg: str):
    log.info(f"{GREEN}{msg}{RESET}")

def gsep():
    log.info(f"{GREEN}{'-'*40}{RESET}")

# -------------------- התקנת FFmpeg אוטומטית --------------------
FFMPEG_EXECUTABLE = "ffmpeg"

def ensure_ffmpeg():
    """
    מוריד FFmpeg סטטי ומתקין מקומית אם לא קיים במערכת
    """
    glog("בודק FFmpeg...")
    global FFMPEG_EXECUTABLE
    if not shutil.which("ffmpeg"):
        glog("FFmpeg לא נמצא, מתקין...")
        ffmpeg_bin_dir = "ffmpeg_bin"
        os.makedirs(ffmpeg_bin_dir, exist_ok=True)
        ffmpeg_url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
        archive_path = os.path.join(ffmpeg_bin_dir, "ffmpeg.tar.xz")
        try:
            r = requests.get(ffmpeg_url, stream=True, timeout=60)
            r.raise_for_status()
            with open(archive_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

            with tarfile.open(archive_path, 'r:xz') as tar_ref:
                tar_ref.extractall(ffmpeg_bin_dir)
            os.remove(archive_path)

            found_ffmpeg_path = None
            for root, _, files in os.walk(ffmpeg_bin_dir):
                if "ffmpeg" in files:
                    found_ffmpeg_path = os.path.join(root, "ffmpeg")
                    break
            if found_ffmpeg_path:
                FFMPEG_EXECUTABLE = found_ffmpeg_path
                os.environ["PATH"] += os.pathsep + os.path.dirname(FFMPEG_EXECUTABLE)
                if os.name == 'posix':
                    os.chmod(FFMPEG_EXECUTABLE, 0o755)
                glog(f"FFmpeg הותקן בהצלחה: {FFMPEG_EXECUTABLE}")
            else:
                log.error("❌ לא נמצא קובץ ffmpeg לאחר חילוץ.")
                FFMPEG_EXECUTABLE = "ffmpeg"
        except Exception as e:
            log.error(f"❌ שגיאה בהתקנת FFmpeg: {e}")
            FFMPEG_EXECUTABLE = "ffmpeg"
    else:
        glog("FFmpeg כבר קיים במערכת.")

# -------------------- הגדרות מערכת --------------------
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- פונקציות עיבוד שמע --------------------
def run_ffmpeg_filter(input_file, output_file, filter_str):
    """הרצה כללית של פילטר FFmpeg"""
    cmd = [FFMPEG_EXECUTABLE, "-y", "-i", input_file, "-af", filter_str, output_file]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"שגיאת FFmpeg: {e}")
        return False

def normalize_pydub(input_file, output_file):
    """יישור עוצמות עם Pydub"""
    try:
        audio = AudioSegment.from_file(input_file, format="wav")
        normalized_audio = effects.normalize(audio)
        normalized_audio.export(output_file, format="wav")
        return True
    except Exception as e:
        log.error(f"שגיאת Normalize: {e}")
        return False

# -------------------- Google Speech Recognition --------------------
def transcribe_google(audio_file, improvement_name, noise_adjustment=False):
    r = sr.Recognizer()
    if noise_adjustment:
        r.energy_threshold = 100
        r.dynamic_energy_threshold = True
        r.pause_threshold = 0.4
        r.non_speaking_duration = 0.1
    try:
        with sr.AudioFile(audio_file) as source:
            if noise_adjustment:
                r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.record(source)
        text = r.recognize_google(audio, language="he-IL")
        log.info(f"🎙️ Google | שיפור: {improvement_name}")
        log.info(f"   → זוהה: {text}")
        return text
    except sr.UnknownValueError:
        log.info(f"🎙️ Google | שיפור: {improvement_name}")
        log.info("   → לא זוהה דיבור ברור.")
        return ""
    except sr.RequestError as e:
        log.error(f"שגיאת Google API: {e}")
        return ""
    except Exception as e:
        log.error(f"שגיאה כללית Google: {e}")
        return ""

# -------------------- Whisper --------------------
def transcribe_whisper(audio_file, improvement_name, model_size="small"):
    try:
        model = whisper.load_model(model_size)
        result = model.transcribe(audio_file, language="he")
        text = result.get("text", "").strip()
        log.info(f"🤖 Whisper | שיפור: {improvement_name}")
        if text:
            log.info(f"   → זוהה: {text}")
        else:
            log.info("   → לא זוהה דיבור ברור.")
        return text
    except Exception as e:
        log.error(f"שגיאת Whisper: {e}")
        return ""

# -------------------- שלבי שיפורים --------------------
def apply_all_improvements(input_path, output_base_dir):
    """
    מייצר את כל קבצי השמע המשופרים לפי השלבים.
    """
    steps = {}

    # 1. קובץ מקורי
    original = os.path.join(output_base_dir, "original.wav")
    AudioSegment.from_file(input_path).export(original, format="wav")
    steps["ללא שיפור"] = original

    # 2. ניקוי תדרים בלבד
    highlow = os.path.join(output_base_dir, "ffmpeg_highlow.wav")
    run_ffmpeg_filter(original, highlow, "highpass=f=200, lowpass=f=3000")
    steps["ניקוי תדרים עם FFmpeg"] = highlow

    # 3. יישור עוצמות בלבד
    dynaudnorm = os.path.join(output_base_dir, "ffmpeg_dynaudnorm.wav")
    run_ffmpeg_filter(original, dynaudnorm, "dynaudnorm")
    steps["יישור עוצמות עם FFmpeg"] = dynaudnorm

    # 4. ניקוי + יישור
    combo = os.path.join(output_base_dir, "ffmpeg_combo.wav")
    run_ffmpeg_filter(original, combo, "highpass=f=200, lowpass=f=3000, dynaudnorm")
    steps["ניקוי תדרים + יישור עוצמות"] = combo

    # 5. Normalize עם Pydub
    normalized = os.path.join(output_base_dir, "pydub_normalized.wav")
    normalize_pydub(combo, normalized)
    steps["נורמליזציה עם Pydub"] = normalized

    # 6. סינון רעשים מתקדם
    noise_reduction = os.path.join(output_base_dir, "noise_reduction.wav")
    run_ffmpeg_filter(normalized, noise_reduction, "afftdn")
    steps["סינון רעשים מתקדם"] = noise_reduction

    # 7. שילוב כל הפילטרים
    final_clean = os.path.join(output_base_dir, "final_clean.wav")
    run_ffmpeg_filter(noise_reduction, final_clean, "highpass=f=200, lowpass=f=3000, dynaudnorm,afftdn")
    steps["שילוב כל הפילטרים"] = final_clean

    return steps

# -------------------- תהליך מלא --------------------
def process_audio(input_file):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    glog(f"📂 התחלת עיבוד חדש - תיקיה: {run_dir}")
    gsep()

    improvements = apply_all_improvements(input_file, run_dir)

    # Google
    log.info("🔹 התחלת זיהוי Google...")
    for name, file_path in improvements.items():
        transcribe_google(file_path, name, noise_adjustment=True)
        log.info(f"   💾 קובץ שמור: {file_path}")
        log.info("-"*50)

    gsep()

    # Whisper
    log.info("🔹 התחלת זיהוי Whisper...")
    for name, file_path in improvements.items():
        transcribe_whisper(file_path, name, model_size="small")
        log.info(f"   💾 קובץ שמור: {file_path}")
        log.info("-"*50)

    gsep()
    log.info("🏁 סיום תהליך הזיהוי")

# -------------------- Flask API --------------------
app = Flask(__name__)

@app.route("/upload_audio", methods=["GET"])
def upload_audio():
    """
    נקודת קצה שמתאימה לבקשות GET.
    חובה לשלוח פרמטר בשם 'path' שמכיל את הנתיב לקובץ שמע בשרת.
    לדוגמה: /upload_audio?path=sample.wav
    """
    audio_path = request.args.get("path")
    if not audio_path or not os.path.exists(audio_path):
        return jsonify({"error": "חסר פרמטר 'path' או שהקובץ לא קיים"}), 400

    glog(f"📡 התקבלה בקשת GET עם קובץ: {audio_path}")
    process_audio(audio_path)
    return jsonify({"status": "ok", "method": "GET"}), 200

# -------------------- הרצה --------------------
if __name__ == "__main__":
    ensure_ffmpeg()
    glog("שרת Flask עלה בכתובת http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
