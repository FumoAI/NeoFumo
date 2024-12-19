from pyannote.audio import Model, Pipeline, Inference
from pyannote.core import Segment
from scipy.spatial.distance import cosine
from config import HUGGINGFACE_TOKEN

def extract_speaker_embedding(pipeline, audio_file, speaker_label):
    diarization = pipeline(audio_file)
    speaker_embedding = None
    for turn, _, label in diarization.itertracks(yield_label=True):
        if label == speaker_label:
            segment = Segment(turn.start + 0.05, turn.end - 0.05) # need fix
            print(f"\n\n\nExtracting speaker embedding for {speaker_label} from {segment.start:.2f}s to {segment.end:.2f}s\n\n\n")
            speaker_embedding = inference.crop(audio_file, segment)
            break
    return speaker_embedding

# 对于给定的音频，提取声纹特征并与人库中的声纹进行比较
def recognize_speaker(pipeline, audio_file):
    diarization = pipeline(audio_file)
    speaker_turns = []
    for turn, _, speaker_label in diarization.itertracks(yield_label=True):
        # 提取切片的声纹特征
        embedding = inference.crop(audio_file, turn)  
        distances = {}
        for speaker, embeddings in speaker_embeddings.items():  
	        # 计算与已知说话人的声纹特征的余弦距离
            distances[speaker] = min([cosine(embedding, e) for e in embeddings])
        # 选择距离最小的说话人
        recognized_speaker = min(distances, key=distances.get)  
        speaker_turns.append((turn, recognized_speaker))  
        # 记录说话人的时间段和余弦距离最小的预测说话人
    return speaker_turns

if __name__ == "__main__":
    token = HUGGINGFACE_TOKEN
    
    # 加载声音分离识别模型
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,  # 在项目页面agree使用协议，并获取 Hugging Face Token
        # cache_dir="/home/huggingface/hub/models--pyannote--speaker-diarization-3.1/"
    )

    # 加载声纹嵌入模型
    embed_model = Model.from_pretrained("pyannote/embedding", use_auth_token=token)
    inference = Inference(embed_model, window="whole")

    # pipeline.to(torch.device("cuda"))

    # 假设您已经有一个包含不同人声的音频文件集,以及对应的人
    person = ["cjl", "jjy", "qjf", "zcy"]
    
    audio_files = { i: f"../audio-demo/{i}.wav" for i in person }

    speaker_embeddings = {}
    for speaker, audio_file in audio_files.items():
        diarization = pipeline(audio_file)
        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            embedding = extract_speaker_embedding(pipeline, audio_file, speaker_label)
            # 获取原始已知说话人的声纹特征
            speaker_embeddings.setdefault(speaker, []).append(embedding)

    # 给定新的未知人物的音频文件
    given_audio_file = "../audio-demo/mix_1.wav"

    # 识别给定音频中的说话人
    recognized_speakers = recognize_speaker(pipeline, given_audio_file)
    print("Recognized speakers in the given audio:")
    for turn, speaker in recognized_speakers:
        print(f"Speaker {speaker} spoke between {turn.start:.2f}s and {turn.end:.2f}s")

