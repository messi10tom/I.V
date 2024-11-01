import sounddevice as sd
import wave
import whisper

# To store recorded audio
recorded_audio = []

# Function to record audio
def record_audio(
        samplerate: int = 44100,  # Sample rate in Hz
        channels: int = 1,        # Number of audio channels
        device_index: int = 5  # Using device index 5, for more info run sd.query_devices()
):
    global recorded_audio
    print("Recording... Press Ctrl+C to stop.")
    try:
        while True:
            # Record for a short period and append to the list
            audio_chunk = sd.rec(int(samplerate * 5), samplerate=samplerate, channels=channels, dtype='float64', device=device_index)
            sd.wait()  # Wait until recording is finished
            recorded_audio.append(audio_chunk)
    except KeyboardInterrupt:
        print("Recording stopped.")

def save_audio(audio, filename, samplerate=44100, channels=1):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16 bits per sample
        wf.setframerate(samplerate)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())

    print(f"Recording saved as '{filename}'.")

