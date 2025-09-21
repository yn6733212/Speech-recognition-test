import os
import tempfile
import logging
import requests
import tarfile
import shutil
from flask import Flask, request, jsonify
from pydub import AudioSegment
import speech_recognition as sr
from rapidfuzz import process, fuzz
import subprocess

# ------------------ Logging Configuration ------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

app = Flask(__name__)

# List of possible keywords to match
KEYWORDS = ["בני ברק", "ירושלים", "תל אביב", "חיפה", "אשדוד"]

# ------------------ FFmpeg Setup ------------------
FFMPEG_EXECUTABLE = "ffmpeg"

def ensure_ffmpeg():
    """
    Ensure FFmpeg is installed and available in the PATH.
    If not found, it will be downloaded and installed locally.
    """
    global FFMPEG_EXECUTABLE

    if shutil.which("ffmpeg"):
        logging.info("FFmpeg is already installed on this system.")
        return

    logging.info("FFmpeg not found. Downloading and installing...")

    ffmpeg_dir = "ffmpeg_bin"
    os.makedirs(ffmpeg_dir, exist_ok=True)

    # Download static build of FFmpeg
    url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
    archive_path = os.path.join(ffmpeg_dir, "ffmpeg.tar.xz")

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(archive_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Extract the downloaded archive
        with tarfile.open(archive_path, "r:xz") as tar_ref:
            tar_ref.extractall(ffmpeg_dir)

        os.remove(archive_path)

        # Find the ffmpeg executable inside the extracted folder
        for root, _, files in os.walk(ffmpeg_dir):
            if "ffmpeg" in files:
                FFMPEG_EXECUTABLE = os.path.join(root, "ffmpeg")
                break

        if FFMPEG_EXECUTABLE:
            os.environ["PATH"] += os.pathsep + os.path.dirname(FFMPEG_EXECUTABLE)
            if os.name == 'posix':  # Linux/Mac
                os.chmod(FFMPEG_EXECUTABLE, 0o755)
            logging.info(f"FFmpeg installed successfully at: {FFMPEG_EXECUTABLE}")
        else:
            logging.error("FFmpeg executable not found after extraction.")

    except Exception as e:
        logging.error(f"Failed to install FFmpeg automatically: {e}")


# ------------------ Helper Functions ------------------

def add_silence(input_path: str) -> AudioSegment:
    """
    Add one second of silence at the beginning and end of the audio file.
    This improves speech recognition accuracy, especially for short recordings.
    """
    logging.info("Adding one second of silence to audio file...")
    audio = AudioSegment.from_file(input_path, format="wav")
    silence = AudioSegment.silent(duration=1000)  # 1000ms = 1 second
    return silence + audio + silence

def recognize_speech(audio_segment: AudioSegment) -> str:
    """
    Perform speech recognition using Google SpeechRecognition API.
    """
    recognizer = sr.Recognizer()
    try:
        # Use a temporary file for SpeechRecognition to read
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
            audio_segment.export(temp_wav.name, format="wav")
            with sr.AudioFile(temp_wav.name) as source:
                data = recognizer.record(source)

            text = recognizer.recognize_google(data, language="he-IL")
            logging.info(f"Recognized text: {text}")
            return text
    except sr.UnknownValueError:
        logging.warning("Speech not detected or unclear.")
        return ""
    except Exception as e:
        logging.error(f"Error during speech recognition: {e}")
        return ""

def find_best_match(text: str) -> str | None:
    """
    Find the closest matching word from the predefined KEYWORDS list.
    """
    if not text:
        return None

    result = process.extractOne(text, KEYWORDS, scorer=fuzz.ratio)
    if result and result[1] >= 80:
        logging.info(f"Best match found: {result[0]} (confidence: {result[1]}%)")
        return result[0]

    logging.info("No sufficient match found.")
    return None

# ------------------ API Endpoint ------------------

@app.route("/upload_audio", methods=["GET"])
def upload_audio():
    """
    Endpoint to receive an audio file via GET parameter,
    download it, process it, and return the recognized text with the best match.
    Example usage:
    /upload_audio?file_url=https://example.com/audio.wav
    """
    file_url = request.args.get("file_url")
    if not file_url:
        logging.error("Missing 'file_url' parameter.")
        return jsonify({"error": "Missing 'file_url' parameter"}), 400

    logging.info(f"Received file URL: {file_url}")

    try:
        # Step 1: Download the audio file
        response = requests.get(file_url, timeout=15)
        if response.status_code != 200:
            logging.error(f"Failed to download audio file. Status code: {response.status_code}")
            return jsonify({"error": "Failed to download audio file"}), 400

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_input:
            temp_input.write(response.content)
            temp_input.flush()
            logging.info(f"Audio downloaded and saved temporarily: {temp_input.name}")

            # Step 2: Add silence
            processed_audio = add_silence(temp_input.name)

            # Step 3: Speech recognition
            recognized_text = recognize_speech(processed_audio)

            # Step 4: Matching against predefined keywords
            matched_word = find_best_match(recognized_text)

            if matched_word:
                logging.info(f"Final matched keyword: {matched_word}")
            else:
                logging.info("No keyword match found.")

    except Exception as e:
        logging.error(f"Processing error: {e}")
        return jsonify({"error": "Error processing the audio file"}), 500

    return jsonify({
        "recognized_text": recognized_text,
        "matched_word": matched_word if matched_word else "No match found"
    })

# ------------------ Run Server ------------------

if __name__ == "__main__":
    # Ensure FFmpeg is installed before running the server
    ensure_ffmpeg()

    port = int(os.environ.get("PORT", 5000))
    logging.info(f"Server running on port {port}")
    app.run(host="0.0.0.0", port=port)
