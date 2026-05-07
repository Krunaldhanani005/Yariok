@echo off
echo Installing Hey Robot dependencies...
pip install SpeechRecognition pyttsx3
pip install pipwin
pipwin install pyaudio
echo.
echo Done! Run the assistant with:
echo   python assistant.py
pause
