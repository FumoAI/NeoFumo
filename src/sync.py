import pyaudio

input_device_index = 0
chunk = 1024
rate = 44100

def record_audio(duration=5):
    audio = pyaudio.PyAudio()

    stream = audio.open(format=pyaudio.paInt16, channels=1,
                        rate=rate, input=True,
                        input_device_index=input_device_index,
                        frames_per_buffer=chunk)
    print("Recording...")

    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        yield data

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

def play_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1,
                        rate=rate, output=True)
    
    
    input_data = record_audio(50)
    for data in input_data:
        stream.write(data)
    
    stream.stop_stream()
    stream.close()
    audio.terminate()

if __name__ == "__main__":
    play_audio()