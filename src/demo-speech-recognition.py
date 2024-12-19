from vosk import Model, KaldiRecognizer
import wave

def transcribe_audio_to_text_vosk(wav_file_path, model_path):
    # 初始化模型
    model = Model(model_path)  # 指定模型目录路径
    # 注意：这里的采样率已经更改为44100
    recognizer = KaldiRecognizer(model, 44100)
    
    # 打开音频文件
    with wave.open(wav_file_path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 44100:
            print("只能输入44.1kHz, 16位, 单声道WAV文件")
            return None
            
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                if result:
                    print("Partial result:", result)
        
        # 获取最终文本
        final_result = recognizer.FinalResult()
        print("Final Transcription: ", final_result)
        return final_result

# 要分析的音频文件路径
input_wav_path = "input_Chinese.wav"
# Kaldi模型解压后的目录路径（确保这个模型是为44.1kHz采样率训练的）
model_path = "./speech-recog/vosk-model-small-cn-0.22"

# 执行识别
transcribed_text = transcribe_audio_to_text_vosk(input_wav_path, model_path)

if transcribed_text: # 测试
    print("Transcribed Text:", transcribed_text)