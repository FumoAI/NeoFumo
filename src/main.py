import pyaudio
import wave

def record_audio(output_filename, duration=5, rate=44100, chunk=1024, input_device_index=None):
    audio = pyaudio.PyAudio()

    # Start recording
    stream = audio.open(format=pyaudio.paInt16, channels=1,
                        rate=rate, input=True,
                        input_device_index=input_device_index,
                        frames_per_buffer=chunk)
    print("Recording...")

    frames = []

    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Finished recording.")

    # Stop recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded data as a WAV file
    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

def play_audio(input_filename, chunk=1024):
    wf = wave.open(input_filename, 'rb')
    audio = pyaudio.PyAudio()

    # Ensure the number of channels is correctly set
    channels = wf.getnchannels()
    if channels not in [1, 2]:
        raise ValueError(f"Invalid number of channels: {channels}")

    # Open stream
    stream = audio.open(format=audio.get_format_from_width(wf.getsampwidth()),
                        channels=channels,
                        rate=wf.getframerate(),
                        output=True)

    # Read data in chunks
    data = wf.readframes(chunk)

    # Play stream
    while data:
        stream.write(data)
        data = wf.readframes(chunk)

    # Stop stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

if __name__ == "__main__":
    # text = "现代标准汉语是以北京官话为基础音，广泛通行于华人地区的语言。现代标准汉语共有四种声调，并在弱音节中使用轻声。"
    output_filename = "zcy.wav"
    input_device_index = 1  # 设置为所需的音频输入设备索引
    record_audio(output_filename, input_device_index=input_device_index)
    # play_audio(output_filename)