from flask import Flask, render_template, request, jsonify, send_from_directory
from gtts import gTTS
import os

app = Flask(__name__)
AUDIO_FOLDER = "static"
AUDIO_FILE = "tts_output.mp3"

@app.route("/")
def index():
    return render_template("tts.html")

@app.route("/speak", methods=["POST"])
def speak():
    text = request.json.get("text")
    if not text:
        return jsonify({"success": False, "error": "No text received"}), 400

    filepath = os.path.join(AUDIO_FOLDER, AUDIO_FILE)
    speech = gTTS(text)
    speech.save(filepath)

    return jsonify({"success": True, "audio_url": f"/static/{AUDIO_FILE}"})

if __name__ == "__main__":
    os.makedirs(AUDIO_FOLDER, exist_ok=True)
    app.run(debug=True)
