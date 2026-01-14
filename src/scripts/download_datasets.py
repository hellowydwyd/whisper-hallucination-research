"""
数据集下载脚本
==============

支持下载以下数据集：
1. ESC-50 - 环境声音分类数据集（推荐，小巧）
2. UrbanSound8K - 城市声音数据集
3. LibriSpeech - 语音识别数据集
4. MUSAN - 音乐、语音、噪声数据集

使用方法:
    conda activate d2l
    python download_datasets.py --dataset esc50       # 下载 ESC-50
    python download_datasets.py --dataset librispeech # 下载 LibriSpeech
    python download_datasets.py --dataset all         # 下载全部
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import argparse
import urllib.request
import zipfile
import tarfile
from tqdm import tqdm
import shutil

from config import DATA_DIR, NON_SPEECH_DIR, SPEECH_DIR


class DownloadProgressBar(tqdm):
    """下载进度条"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path, desc: str = "Downloading"):
    """
    下载文件并显示进度
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)
    
    return output_path


def download_esc50():
    """
    下载 ESC-50 环境声音分类数据集
    
    - 50 类环境声音
    - 2000 个音频样本（每类 40 个）
    - 5 秒 / 样本
    - 非常适合测试 Whisper 幻觉
    
    GitHub: https://github.com/karolpiczak/ESC-50
    """
    print("\n" + "=" * 50)
    print("下载 ESC-50 数据集")
    print("=" * 50)
    
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    output_dir = NON_SPEECH_DIR / "esc50"
    zip_path = DATA_DIR / "esc50.zip"
    
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"[ESC-50] 数据集已存在: {output_dir}")
        return output_dir
    
    print(f"[ESC-50] 下载中... (约 600MB)")
    print(f"[ESC-50] URL: {url}")
    
    try:
        download_file(url, zip_path, "ESC-50")
        
        print(f"[ESC-50] 解压中...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        
        # 移动到目标目录
        extracted_dir = DATA_DIR / "ESC-50-master"
        if extracted_dir.exists():
            shutil.move(str(extracted_dir), str(output_dir))
        
        # 清理
        zip_path.unlink()
        
        print(f"[ESC-50] 下载完成: {output_dir}")
        
        # 统计
        audio_files = list((output_dir / "audio").glob("*.wav")) if (output_dir / "audio").exists() else []
        print(f"[ESC-50] 音频文件数: {len(audio_files)}")
        
        return output_dir
        
    except Exception as e:
        print(f"[ESC-50] 下载失败: {e}")
        print("[ESC-50] 请手动下载: https://github.com/karolpiczak/ESC-50")
        return None


def download_urbansound8k():
    """
    下载 UrbanSound8K 数据集
    
    - 10 类城市声音（狗叫、警笛、引擎等）
    - 8732 个音频样本
    - 需要在官网注册下载
    
    官网: https://urbansounddataset.weebly.com/urbansound8k.html
    """
    print("\n" + "=" * 50)
    print("下载 UrbanSound8K 数据集")
    print("=" * 50)
    
    output_dir = NON_SPEECH_DIR / "urbansound8k"
    
    print("[UrbanSound8K] ⚠️ 此数据集需要在官网注册后下载")
    print("[UrbanSound8K] 下载链接: https://urbansounddataset.weebly.com/urbansound8k.html")
    print(f"[UrbanSound8K] 请下载后解压到: {output_dir}")
    print("\n[UrbanSound8K] 数据集包含以下类别:")
    categories = [
        "air_conditioner", "car_horn", "children_playing", "dog_bark", 
        "drilling", "engine_idling", "gun_shot", "jackhammer", 
        "siren", "street_music"
    ]
    for i, cat in enumerate(categories):
        print(f"  {i}: {cat}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def download_librispeech():
    """
    下载 LibriSpeech test-clean 数据集
    
    - 英语语音识别标准测试集
    - 用于计算 WER
    
    官网: https://www.openslr.org/12/
    """
    print("\n" + "=" * 50)
    print("下载 LibriSpeech test-clean 数据集")
    print("=" * 50)
    
    url = "https://www.openslr.org/resources/12/test-clean.tar.gz"
    output_dir = SPEECH_DIR / "librispeech"
    tar_path = DATA_DIR / "test-clean.tar.gz"
    
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"[LibriSpeech] 数据集已存在: {output_dir}")
        return output_dir
    
    print(f"[LibriSpeech] 下载中... (约 350MB)")
    
    try:
        download_file(url, tar_path, "LibriSpeech test-clean")
        
        print(f"[LibriSpeech] 解压中...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(DATA_DIR)
        
        # 移动到目标目录
        extracted_dir = DATA_DIR / "LibriSpeech" / "test-clean"
        if extracted_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(extracted_dir), str(output_dir / "test-clean"))
            # 清理空目录
            (DATA_DIR / "LibriSpeech").rmdir()
        
        # 清理
        tar_path.unlink()
        
        print(f"[LibriSpeech] 下载完成: {output_dir}")
        
        return output_dir
        
    except Exception as e:
        print(f"[LibriSpeech] 下载失败: {e}")
        print("[LibriSpeech] 请手动下载: https://www.openslr.org/12/")
        return None


def download_musan():
    """
    下载 MUSAN 数据集（音乐、语音、噪声）
    
    - music: 音乐片段
    - speech: 语音片段  
    - noise: 噪声片段
    
    官网: https://www.openslr.org/17/
    """
    print("\n" + "=" * 50)
    print("下载 MUSAN 数据集")
    print("=" * 50)
    
    url = "https://www.openslr.org/resources/17/musan.tar.gz"
    output_dir = NON_SPEECH_DIR / "musan"
    tar_path = DATA_DIR / "musan.tar.gz"
    
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"[MUSAN] 数据集已存在: {output_dir}")
        return output_dir
    
    print(f"[MUSAN] 下载中... (约 11GB，请耐心等待)")
    print(f"[MUSAN] 如果下载太慢，可以手动下载: {url}")
    
    try:
        download_file(url, tar_path, "MUSAN")
        
        print(f"[MUSAN] 解压中...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(DATA_DIR)
        
        # 移动到目标目录
        extracted_dir = DATA_DIR / "musan"
        if extracted_dir.exists() and extracted_dir != output_dir:
            shutil.move(str(extracted_dir), str(output_dir))
        
        # 清理
        if tar_path.exists():
            tar_path.unlink()
        
        print(f"[MUSAN] 下载完成: {output_dir}")
        
        return output_dir
        
    except Exception as e:
        print(f"[MUSAN] 下载失败: {e}")
        print("[MUSAN] 请手动下载: https://www.openslr.org/17/")
        return None


def download_from_huggingface():
    """
    使用 Hugging Face datasets 库下载数据集
    这种方式更稳定，推荐使用
    """
    print("\n" + "=" * 50)
    print("使用 Hugging Face 下载数据集")
    print("=" * 50)
    
    try:
        from datasets import load_dataset
        
        # 下载 LibriSpeech
        print("\n[HuggingFace] 下载 LibriSpeech test-clean...")
        dataset = load_dataset(
            "librispeech_asr", 
            "clean", 
            split="test",
            cache_dir=str(DATA_DIR / "hf_cache")
        )
        print(f"[HuggingFace] LibriSpeech 样本数: {len(dataset)}")
        
        # 保存一些样本用于测试
        output_dir = SPEECH_DIR / "librispeech_samples"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[HuggingFace] 保存样本到: {output_dir}")
        
        import soundfile as sf
        for i, sample in enumerate(dataset.select(range(min(100, len(dataset))))):
            audio = sample['audio']
            sf.write(
                str(output_dir / f"sample_{i:04d}.wav"),
                audio['array'],
                audio['sampling_rate']
            )
            
            # 保存转录文本
            with open(output_dir / f"sample_{i:04d}.txt", 'w') as f:
                f.write(sample['text'])
        
        print(f"[HuggingFace] 已保存 {min(100, len(dataset))} 个样本")
        
        return output_dir
        
    except ImportError:
        print("[HuggingFace] 请先安装: pip install datasets")
        return None
    except Exception as e:
        print(f"[HuggingFace] 下载失败: {e}")
        return None


def create_synthetic_test_data():
    """
    创建合成测试数据（用于快速测试）
    
    生成各种类型的非语音音频：
    - 静音
    - 白噪声
    - 粉红噪声
    - 正弦波
    """
    print("\n" + "=" * 50)
    print("创建合成测试数据")
    print("=" * 50)
    
    from utils.audio_utils import generate_silence, generate_noise, save_audio
    import numpy as np
    
    output_dir = NON_SPEECH_DIR / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    durations = [1, 5, 10, 20, 30]
    
    print("[Synthetic] 生成静音音频...")
    for duration in durations:
        audio = generate_silence(duration)
        save_audio(audio, output_dir / f"silence_{duration}s.wav")
    
    print("[Synthetic] 生成白噪声音频...")
    for duration in durations:
        for i in range(3):
            audio = generate_noise(duration, noise_type="white")
            save_audio(audio, output_dir / f"white_noise_{duration}s_{i}.wav")
    
    print("[Synthetic] 生成粉红噪声音频...")
    for duration in durations:
        for i in range(3):
            audio = generate_noise(duration, noise_type="pink")
            save_audio(audio, output_dir / f"pink_noise_{duration}s_{i}.wav")
    
    print("[Synthetic] 生成正弦波音频...")
    sr = 16000
    for freq in [100, 440, 1000, 5000]:
        for duration in [5, 10]:
            t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
            audio = 0.5 * np.sin(2 * np.pi * freq * t)
            save_audio(audio, output_dir / f"sine_{freq}hz_{duration}s.wav")
    
    # 统计
    audio_files = list(output_dir.glob("*.wav"))
    print(f"\n[Synthetic] 完成! 共生成 {len(audio_files)} 个音频文件")
    print(f"[Synthetic] 保存位置: {output_dir}")
    
    return output_dir


def print_dataset_summary():
    """
    打印数据集摘要
    """
    print("\n" + "=" * 60)
    print("📊 数据集摘要")
    print("=" * 60)
    
    # 检查各目录
    datasets = [
        ("ESC-50", NON_SPEECH_DIR / "esc50"),
        ("UrbanSound8K", NON_SPEECH_DIR / "urbansound8k"),
        ("MUSAN", NON_SPEECH_DIR / "musan"),
        ("Synthetic", NON_SPEECH_DIR / "synthetic"),
        ("LibriSpeech", SPEECH_DIR / "librispeech"),
        ("HuggingFace Samples", SPEECH_DIR / "librispeech_samples"),
    ]
    
    for name, path in datasets:
        if path.exists():
            # 计算音频文件数
            audio_count = sum(1 for _ in path.rglob("*.wav"))
            audio_count += sum(1 for _ in path.rglob("*.flac"))
            audio_count += sum(1 for _ in path.rglob("*.mp3"))
            
            if audio_count > 0:
                print(f"  ✓ {name}: {audio_count} 个音频文件")
            else:
                print(f"  ○ {name}: 目录存在但无音频")
        else:
            print(f"  ✗ {name}: 未下载")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='下载测试数据集')
    parser.add_argument(
        '--dataset',
        type=str,
        default='synthetic',
        choices=['esc50', 'urbansound8k', 'librispeech', 'musan', 'huggingface', 'synthetic', 'all'],
        help='要下载的数据集'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("   Whisper ASR 幻觉研究 - 数据集下载")
    print("=" * 60)
    
    if args.dataset == 'esc50':
        download_esc50()
    elif args.dataset == 'urbansound8k':
        download_urbansound8k()
    elif args.dataset == 'librispeech':
        download_librispeech()
    elif args.dataset == 'musan':
        download_musan()
    elif args.dataset == 'huggingface':
        download_from_huggingface()
    elif args.dataset == 'synthetic':
        create_synthetic_test_data()
    elif args.dataset == 'all':
        create_synthetic_test_data()  # 先创建合成数据
        download_esc50()
        download_librispeech()
        # download_musan()  # 太大，默认不下载
    
    print_dataset_summary()
    
    print("\n✅ 数据集准备完成!")
    print("现在可以运行实验: python run_experiment.py --mode quick")


if __name__ == "__main__":
    main()
