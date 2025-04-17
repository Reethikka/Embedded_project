# speech-to-text
Speech-to-text program developed for use with Raspberry Pi

the file 1 is a terminal output program using the SpeechRecognition library. 
it requires internet access.



the file 2 is a tkinter gui program that uses python's SpeechRecognition library. 
it requires internet access.

sudo apt update
sudo apt install python3-pip python3-tkportaudio
pip3 install SpeechRecognition pyaudio

if pyaudio gives errors:
sudo apt install portaudio19-dev
pip3 install pyaudio



the file speech-to-text uses speechrecognition and requires internet access.
it uses flask and socketio for backend with a simple html frontend

pip install flask flask-socketio speechrecognition

run server and go to localhost:5000



the file text-to-speech uses gtts and requires internet access. it uses flask and socketio with a html frontend.

pip install gTTS 

this can be run similar to speech-to-text