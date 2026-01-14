"""
Whisper ASR 幻觉研究 - 配置文件
"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"

# 创建必要的目录
for dir_path in [DATA_DIR, OUTPUT_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 数据子目录
RAW_AUDIO_DIR = DATA_DIR / "raw"           # 原始音频
SPEECH_DIR = DATA_DIR / "speech"           # 语音数据
NON_SPEECH_DIR = DATA_DIR / "non_speech"   # 非语音数据
PROCESSED_DIR = DATA_DIR / "processed"     # 处理后数据

for dir_path in [RAW_AUDIO_DIR, SPEECH_DIR, NON_SPEECH_DIR, PROCESSED_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Whisper 模型配置
WHISPER_MODEL_SIZE = "large-v3"  # tiny, base, small, medium, large, large-v2, large-v3
WHISPER_DEVICE = "cuda"      # cuda 或 cpu
WHISPER_LANGUAGE = "en"      # 识别语言，None 为自动检测

# 音频配置
SAMPLE_RATE = 16000          # Whisper 要求的采样率
MAX_AUDIO_LENGTH = 30        # 最大音频长度（秒）

# 实验配置
RANDOM_SEED = 42

# 幻觉检测配置
HALLUCINATION_THRESHOLD = 0.5  # 幻觉判定阈值
BOH_NGRAM_THRESHOLD = -10      # BoH n-gram 对数概率阈值
BOH_MIN_COUNT = 4              # BoH 最小出现次数

# 常见幻觉短语 (Bag of Hallucinations)
COMMON_HALLUCINATIONS = [
    "thank you",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "like and subscribe",
    "subtitles by the amara org community",
    "i'm not sure what i'm doing here",
    "you",
    "the",
    "so",
]

print(f"[Config] 项目根目录: {PROJECT_ROOT}")
print(f"[Config] 数据目录: {DATA_DIR}")
print(f"[Config] 输出目录: {OUTPUT_DIR}")
