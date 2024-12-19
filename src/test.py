from pyannote.audio import Pipeline
from config import HUGGINGFACE_TOKEN

# 使用您的 Hugging Face 访问令牌
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HUGGINGFACE_TOKEN)
 
# 可选：将管道发送到 GPU（如果有）
# import torch
# pipeline.to(torch.device("cuda"))
 
# 应用预训练管道
diarization = pipeline("../audio-demo/mix_2.wav")
 
# 打印结果
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")