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
    log.info(f"{GREEN}{'-'*50}{RESET}")

# -------------------- הגדרות כלליות --------------------
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOKEN = "0733181201:6714453"  # פרטי ימות
YEMOT_DOWNLOAD_URL = "https://www.call2all.co.il/ym/api/DownloadFile"
BASE_URL = "https://speech-recognition-test-production.up.railway.app"

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

# -------------------- פונקציות בסיסיות --------------------
def run_ffmpeg_filter(input_file, output_file, filter_str):
    """הרצת פילטר FFmpeg"""
    cmd = [FFMPEG_EXECUTABLE, "-y", "-i", input_file, "-af", filter_str, output_file]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def normalize_pydub(input_file, output_file):
    """יישור עוצמות עם Pydub"""
    audio = AudioSegment.from_file(input_file, format="wav")
    normalized_audio = effects.normalize(audio)
    normalized_audio.export(output_file, format="wav")

def add_silence(input_file, output_file, ms=500):
    """הוספת שקט מלאכותי בהתחלה ובסוף"""
    sound = AudioSegment.from_file(input_file, format="wav")
    silence = AudioSegment.silent(duration=ms)
    padded = silence + sound + silence
    padded.export(output_file, format="wav")
    return output_file

# -------------------- זיהוי דיבור של Google --------------------
def transcribe_google(audio_file, file_name):
    r = sr.Recognizer()
    r.energy_threshold = 150
    r.dynamic_energy_threshold = True
    r.pause_threshold = 0.8
    r.non_speaking_duration = 0.3
    try:
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source)
        text = r.recognize_google(audio, language="he-IL")
        log.info(f"🎙️ Google | {file_name} → זוהה: {text}")
        return text
    except sr.UnknownValueError:
        log.info(f"🎙️ Google | {file_name} → לא זוהה דיבור ברור")
        return ""
    except Exception as e:
        log.error(f"❌ שגיאה בזיהוי Google ({file_name}): {e}")
        return ""

# -------------------- שיפורי שמע --------------------
def apply_enhancement(input_file, output_file, enhancement_type, strength="weak"):
    """
    enhancement_type יכול להיות:
    highpass_lowpass, noise_reduction, dynaudnorm, compressor, reencode
    """
    if enhancement_type == "highpass_lowpass":
        if strength == "weak":
            filter_str = "highpass=f=200, lowpass=f=3000"
        else:
            filter_str = "highpass=f=400, lowpass=f=2500"
        run_ffmpeg_filter(input_file, output_file, filter_str)

    elif enhancement_type == "noise_reduction":
        if strength == "weak":
            filter_str = "afftdn=nf=-25"
        else:
            filter_str = "afftdn=nf=-35"
        run_ffmpeg_filter(input_file, output_file, filter_str)

    elif enhancement_type == "dynaudnorm":
        if strength == "weak":
            filter_str = "dynaudnorm=f=250:g=15"
        else:
            filter_str = "dynaudnorm=f=500:g=31"
        run_ffmpeg_filter(input_file, output_file, filter_str)

    elif enhancement_type == "compressor":
        if strength == "weak":
            filter_str = "acompressor=threshold=-15dB:ratio=2:attack=20:release=250"
        else:
            filter_str = "acompressor=threshold=-25dB:ratio=4:attack=10:release=100"
        run_ffmpeg_filter(input_file, output_file, filter_str)

    elif enhancement_type == "reencode":
        cmd = [FFMPEG_EXECUTABLE, "-y", "-i", input_file, "-ar", "16000", "-ac", "1", output_file]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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

# -------------------- עיבוד מלא --------------------
def process_audio(input_file):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    glog(f"📂 התחלת עיבוד חדש - תיקיה: {run_dir}")
    gsep()

    results = []

    # 1. שמירה של הקובץ המקורי
    original = os.path.join(run_dir, "01_ללא_שיפורים_ללא_שקט.wav")
    AudioSegment.from_file(input_file).export(original, format="wav")

    # 2. גרסה עם שקט
    original_silence = os.path.join(run_dir, "02_ללא_שיפורים_עם_שקט.wav")
    add_silence(original, original_silence)

    results.append((original, "ללא שיפורים ללא שקט"))
    results.append((original_silence, "ללא שיפורים עם שקט"))

    # 3. כל סוגי השיפורים
    enhancements = [
        ("highpass_lowpass", "סינון תדרים"),
        ("noise_reduction", "סינון רעשים"),
        ("dynaudnorm", "יישור עוצמות"),
        ("compressor", "קומפרסור"),
        ("reencode", "שינוי קידוד")
    ]

    for enh_type, enh_name in enhancements:
        for strength in ["weak", "strong"]:
            strength_name = "חלש" if strength == "weak" else "חזק"
            # קובץ ללא שקט
            file_no_silence = os.path.join(run_dir, f"{enh_name}_{strength_name}_ללא_שקט.wav")
            apply_enhancement(original, file_no_silence, enh_type, strength)
            results.append((file_no_silence, f"{enh_name} {strength_name} ללא שקט"))

            # קובץ עם שקט
            file_with_silence = os.path.join(run_dir, f"{enh_name}_{strength_name}_עם_שקט.wav")
            temp_silent = os.path.join(run_dir, "temp_silent.wav")
            add_silence(original, temp_silent)
            apply_enhancement(temp_silent, file_with_silence, enh_type, strength)
            results.append((file_with_silence, f"{enh_name} {strength_name} עם שקט"))
            os.remove(temp_silent)

    # 4. כל השיפורים יחד (חלש וחזק)
    for strength in ["weak", "strong"]:
        strength_name = "חלש" if strength == "weak" else "חזק"
        combined = os.path.join(run_dir, f"כל_השיפורים_{strength_name}_ללא_שקט.wav")
        temp_file = original
        for enh_type, _ in enhancements:
            next_temp = os.path.join(run_dir, f"temp_{enh_type}.wav")
            apply_enhancement(temp_file, next_temp, enh_type, strength)
            temp_file = next_temp
        shutil.copy(temp_file, combined)
        results.append((combined, f"כל השיפורים {strength_name} ללא שקט"))

        combined_silence = os.path.join(run_dir, f"כל_השיפורים_{strength_name}_עם_שקט.wav")
        add_silence(combined, combined_silence)
        results.append((combined_silence, f"כל השיפורים {strength_name} עם שקט"))

    # 5. זיהוי דיבור ושמירת תוצאות
    results_txt = os.path.join(run_dir, "תוצאות_זיהוי_דיבור.txt")
    with open(results_txt, "w", encoding="utf-8") as f:
        for file_path, description in results:
            text = transcribe_google(file_path, description)
            f.write(f"{description}: {text}\n")

    # 6. יצירת קובץ ZIP
    zip_path = create_zip_from_folder(run_dir)
    glog(f"📦 קובץ ZIP נוצר: {zip_path}")
    glog(f"📥 להורדה: {BASE_URL}/download/{os.path.basename(zip_path)}")
    gsep()

    return zip_path

# -------------------- Flask API --------------------
app = Flask(__name__)

@app.route("/upload_audio", methods=["GET"])
def upload_audio():
    stockname = request.args.get('stockname')
    if not stockname:
        return jsonify({"error": "חסר פרמטר 'stockname'"}), 400

    yemot_path = f"ivr2:{stockname}"
    params = {"token": TOKEN, "path": yemot_path}
    glog(f"📡 התקבלה הקלטה מהשלוחה: {stockname}")
    try:
        response = requests.get(YEMOT_DOWNLOAD_URL, params=params, timeout=30)
        response.raise_for_status()
        temp_file = "temp_input.wav"
        with open(temp_file, 'wb') as f:
            f.write(response.content)

        zip_path = process_audio(temp_file)

        return send_file(zip_path, as_attachment=True)

    except Exception as e:
        log.error(f"שגיאה בהורדת קובץ מימות: {e}")
        return jsonify({"error": "שגיאה בהורדה או בעיבוד הקובץ"}), 500

# נתיב להורדה ישירה
@app.route("/download/<filename>", methods=["GET"])
def download_file(filename):
    full_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(full_path):
        return jsonify({"error": "קובץ לא נמצא"}), 404
    return send_file(full_path, as_attachment=True)

# -------------------- הרצה --------------------
if __name__ == "__main__":
    ensure_ffmpeg()
    glog("שרת Flask עלה בכתובת http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
