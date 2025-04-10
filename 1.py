import speech_recognition as sr

# Initialize recognizer
r = sr.Recognizer()

# Use microphone as source
with sr.Microphone() as source:
    print("Speak something...")
    audio = r.listen(source)

    try:
        # Recognize using Google Web Speech API
        text = r.recognize_google(audio)
        print("You said:", text)

    except sr.UnknownValueError:
        print("Could not understand audio.")
    except sr.RequestError:
        print("Could not request results; check your internet connection.")
