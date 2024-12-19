import pyaudio
import wave
import audioop
import pyttsx3
from vosk import Model, KaldiRecognizer
import openai
import json
from openai import OpenAI
from config import BASE_URL, API_KEY

def record_audio_with_silence_detection(rate=44100, chunk=1024, silence_threshold=1000, silence_duration=1, input_device_index=None):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1,
                        rate=rate, input=True,
                        input_device_index=input_device_index,
                        frames_per_buffer=chunk)
    print("Recording...")
    _ = input()
    frames = []
    silent_chunks = 0
    while True:
        data = stream.read(chunk)
        frames.append(data)
        rms = audioop.rms(data, 2)
        if rms < silence_threshold:
            silent_chunks += 1
        else:
            silent_chunks = 0
        if silent_chunks > (rate / chunk * silence_duration):
            break
    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    return b''.join(frames)

def transcribe_audio_to_text_vosk(audio_data, model_path, rate=44100):
    model = Model(model_path)
    recognizer = KaldiRecognizer(model, rate)
    recognizer.AcceptWaveform(audio_data)
    result = recognizer.FinalResult()
    text = json.loads(result).get('text', '')
    return text

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY
)

def get_ai_response(prompt):    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{ "role": "user", "content": prompt }]
    )
    reply = completion.choices[0].message.content
    return reply

def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 180)
    engine.setProperty('volume', 1.0)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.say(text)
    engine.runAndWait()
    engine.stop()

if __name__ == "__main__":
    while True:
        input_device_index = None
        audio_data = record_audio_with_silence_detection(silence_threshold=500, silence_duration=1, input_device_index=input_device_index)
        model_path = "../speech-recog/vosk-model-small-en-us-0.15"
        transcribed_text = transcribe_audio_to_text_vosk(audio_data, model_path)
        print("User said:", transcribed_text)
        if not transcribed_text:
            continue
        ai_response = get_ai_response(transcribed_text)
        print("AI:", ai_response)
        speak_text(ai_response)