import os
import sys
import logging
import datetime
import subprocess
import requests
import shutil
import tarfile
import zipfile
from flask import Flask, request, jsonify, send_file
from pydub import AudioSegment, effects
import speech_recognition as sr
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
    out_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(out_handler)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore")

setup_logging()
log = logging.getLogger(__name__)

GREEN = "\033[92m"
RESET = "\033[0m"

def glog(msg: str):
    log.info(f"{GREEN}{msg}{RESET}")

def gsep():
    log.info(f"{GREEN}{'-'*40}{RESET}")

# -------------------- התקנת FFmpeg --------------------
FFMPEG_EXECUTABLE = "ffmpeg"

def ensure_ffmpeg():
    """מוודא ש-FFmpeg קיים, ואם לא - מתקין"""
    glog("בודק FFmpeg...")
    global FFMPEG_EXECUTABLE
    if not shutil.which("ffmpeg"):
        glog("FFmpeg לא נמצא, מתקין...")
        ffmpeg_bin_dir = "ffmpeg_bin"
        os.makedirs(ffmpeg_bin_dir, exist_ok=True)
        url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
        archive_path = os.path.join(ffmpeg_bin_dir, "ffmpeg.tar.xz")
        try:
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()
            with open(archive_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            with tarfile.open(archive_path, 'r:xz') as tar_ref:
                tar_ref.extractall(ffmpeg_bin_dir)
            os.remove(archive_path)

            for root, _, files in os.walk(ffmpeg_bin_dir):
                if "ffmpeg" in files:
                    FFMPEG_EXECUTABLE = os.path.join(root, "ffmpeg")
                    break

            if FFMPEG_EXECUTABLE:
                os.environ["PATH"] += os.pathsep + os.path.dirname(FFMPEG_EXECUTABLE)
                if os.name == 'posix':
                    os.chmod(FFMPEG_EXECUTABLE, 0o755)
                glog(f"FFmpeg הותקן בהצלחה: {FFMPEG_EXECUTABLE}")
            else:
                log.error("❌ לא נמצא קובץ FFmpeg לאחר חילוץ.")
        except Exception as e:
            log.error(f"❌ שגיאה בהתקנת FFmpeg: {e}")
    else:
        glog("FFmpeg כבר קיים במערכת.")

# -------------------- הגדרות כלליות --------------------
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOKEN = "0733181201:6714453"  # פרטי ימות
YEMOT_DOWNLOAD_URL = "https://www.call2all.co.il/ym/api/DownloadFile"

# -------------------- פונקציות עיבוד שמע --------------------
def run_ffmpeg_filter(input_file, output_file, filter_str):
    """הרצת פילטר FFmpeg"""
    cmd = [FFMPEG_EXECUTABLE, "-y", "-i", input_file, "-af", filter_str, output_file]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def normalize_pydub(input_file, output_file):
    """יישור עוצמות בסיסי"""
    audio = AudioSegment.from_file(input_file, format="wav")
    normalized_audio = effects.normalize(audio)
    normalized_audio.export(output_file, format="wav")

# -------------------- Google Speech Recognition --------------------
def transcribe_google(audio_file, improvement_name):
    """זיהוי דיבור בגוגל"""
    r = sr.Recognizer()
    r.energy_threshold = 150
    r.dynamic_energy_threshold = True
    r.pause_threshold = 0.8           # ממתין קצת יותר לפני סיום
    r.non_speaking_duration = 0.3     # זמן שקט מינימלי
    try:
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source)
        text = r.recognize_google(audio, language="he-IL")
        log.info(f"🎙️ Google | {improvement_name} → זוהה: {text}")
        return text
    except sr.UnknownValueError:
        log.info(f"🎙️ Google | {improvement_name} → לא זוהה דיבור ברור")
        return ""
    except Exception as e:
        log.error(f"שגיאה כללית בזיהוי Google: {e}")
        return ""

# -------------------- יצירת ZIP --------------------
def create_zip_from_folder(folder_path):
    zip_name = folder_path + ".zip"
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=folder_path)
                zipf.write(file_path, arcname)
    return zip_name

# -------------------- תהליך מלא --------------------
def process_audio(input_file):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    glog(f"📂 התחלת עיבוד חדש - תיקיה: {run_dir}")
    gsep()

    # --- שמות קבצים בעברית ---
    original = os.path.join(run_dir, "זיהוי דיבור - ללא שיפורים.wav")
    weak = os.path.join(run_dir, "זיהוי דיבור - עם שיפורים חלשים.wav")
    strong = os.path.join(run_dir, "זיהוי דיבור - עם שיפורים חזקים.wav")

    # שלב 1: שמירת הקובץ המקורי
    AudioSegment.from_file(input_file).export(original, format="wav")

    # שלב 2: שיפורים חלשים
    run_ffmpeg_filter(original, weak, "highpass=f=200, lowpass=f=3000")

    # שלב 3: שיפורים חזקים
    run_ffmpeg_filter(weak, strong, "highpass=f=300, lowpass=f=3400, dynaudnorm,afftdn,volume=1.3")

    # שלב 4: הרצת זיהוי גוגל על כל הגרסאות
    files_to_check = {
        "ללא שיפורים": original,
        "שיפורים חלשים": weak,
        "שיפורים חזקים": strong
    }

    for name, path in files_to_check.items():
        transcribe_google(path, name)
        log.info("-"*50)

    # שלב 5: יצירת קובץ ZIP
    zip_path = create_zip_from_folder(run_dir)
    glog(f"📦 קובץ ZIP מוכן להורדה: /download/{timestamp}")
    gsep()

    return zip_path, timestamp

# -------------------- Flask API --------------------
app = Flask(__name__)

@app.route("/upload_audio", methods=["GET"])
def upload_audio():
    stockname = request.args.get('stockname')
    if not stockname:
        return jsonify({"error": "חסר פרמטר 'stockname'"}), 400

    # הורדת הקובץ מימות
    yemot_path = f"ivr2:{stockname}"
    params = {"token": TOKEN, "path": yemot_path}
    glog(f"📡 התקבלה הקלטה מהשלוחה: {stockname}")
    try:
        response = requests.get(YEMOT_DOWNLOAD_URL, params=params, timeout=30)
        response.raise_for_status()
        temp_file = "temp_input.wav"
        with open(temp_file, 'wb') as f:
            f.write(response.content)

        # עיבוד ההקלטה
        zip_path, timestamp = process_audio(temp_file)

        # מחזיר קישור ישיר להורדה
        download_url = f"/download/{timestamp}"
        return jsonify({
            "status": "ok",
            "download_link": download_url
        })

    except Exception as e:
        log.error(f"שגיאה בהורדת קובץ מימות: {e}")
        return jsonify({"error": "שגיאה בהורדה או בעיבוד הקובץ"}), 500

@app.route("/download/<timestamp>", methods=["GET"])
def download_zip(timestamp):
    zip_path = os.path.join(OUTPUT_DIR, f"{timestamp}.zip")
    if not os.path.exists(zip_path):
        return jsonify({"error": "קובץ לא נמצא"}), 404
    return send_file(zip_path, as_attachment=True, download_name=f"processed_{timestamp}.zip")

# -------------------- הרצה --------------------
if __name__ == "__main__":
    ensure_ffmpeg()
    glog("שרת Flask עלה בכתובת http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
