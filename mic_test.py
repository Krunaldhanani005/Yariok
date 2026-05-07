"""Run this to diagnose microphone issues. Run separately from smart_system.py."""
import speech_recognition as sr
import pyaudio

print("=" * 50)
print("  MICROPHONE DIAGNOSTIC")
print("=" * 50)

# List all available mics
print("\nAll microphones found:")
mics = sr.Microphone.list_microphone_names()
for i, name in enumerate(mics):
    print(f"  [{i}] {name}")

# Show PyAudio default
try:
    p = pyaudio.PyAudio()
    info = p.get_default_input_device_info()
    print(f"\nDefault input: [{info['index']}] {info['name']}")
    p.terminate()
except Exception as e:
    print(f"\nPyAudio error: {e}")

# Test recording
print("\n" + "=" * 50)
print("  SPEAK NOW — testing for 6 seconds...")
print("=" * 50)

r = sr.Recognizer()
r.energy_threshold = 100       # very low so it catches anything
r.dynamic_energy_threshold = False

try:
    with sr.Microphone() as src:
        print(f"Mic open. Threshold = {r.energy_threshold}. Speak anything...")
        try:
            audio = r.listen(src, timeout=6, phrase_time_limit=6)
            print("Audio captured! Sending to Google...")
            try:
                text = r.recognize_google(audio)
                print(f"\n  RECOGNIZED: {text}")
            except sr.UnknownValueError:
                print("\n  Audio captured but speech not understood.")
                print("  → Speak louder / closer to mic")
            except sr.RequestError as e:
                print(f"\n  Google STT error: {e}")
                print("  → Check internet connection")
        except sr.WaitTimeoutError:
            print("\n  NO AUDIO in 6 seconds!")
            print("  → Wrong microphone selected OR mic is muted")
            print("  → Go to Windows Sound Settings > Input > check your mic is set as default")
except OSError as e:
    print(f"\nCannot open microphone: {e}")
    print("→ PyAudio / microphone driver issue")

print("\nDone.")
