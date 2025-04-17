from flask import Flask, render_template
from flask_socketio import SocketIO
import speech_recognition as sr
import threading

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

def listen_and_emit():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
    while True:
        try:
            with mic as source:
                audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio)
            print("Recognized:", text)
            socketio.emit('speech_text', {'text': text})
        except Exception as e:
            print("Error:", e)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    threading.Thread(target=listen_and_emit, daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5000)
