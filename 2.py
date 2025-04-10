import speech_recognition as sr
import tkinter as tk
from threading import Thread

# Initialize recognizer
r = sr.Recognizer()

# Create GUI window
root = tk.Tk()
root.title("Live Speech to Text")
root.geometry("600x400")

# Create text widget
text_widget = tk.Text(root, wrap=tk.WORD, font=("Helvetica", 14))
text_widget.pack(expand=True, fill=tk.BOTH)

# Function to update text in GUI
def update_text(transcribed_text):
    text_widget.insert(tk.END, transcribed_text + "\n")
    text_widget.see(tk.END)  

# Speech recognition in a loop
def listen_loop():
    with sr.Microphone() as source:
        while True:
            try:
                print("🎤 Listening...")
                audio = r.listen(source)
                text = r.recognize_google(audio)
                print("📝 You said:", text)
                update_text(text)
            except sr.UnknownValueError:
                update_text("[Unrecognized Speech]")
            except sr.RequestError:
                update_text("[API Request Error – Check Internet]")

# Start the listener thread
listener_thread = Thread(target=listen_loop)
listener_thread.daemon = True
listener_thread.start()

# Run the GUI
try:
    root.mainloop()
except KeyboardInterrupt:
    print("❌ Program interrupted by user.")
