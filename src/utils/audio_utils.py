"""
音频处理工具函数
"""
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Union, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


def load_audio(
    file_path: Union[str, Path], 
    target_sr: int = 16000,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    加载音频文件并转换为指定采样率
    
    Args:
        file_path: 音频文件路径
        target_sr: 目标采样率 (Whisper 要求 16000)
        mono: 是否转换为单声道
        
    Returns:
        audio: 音频数据 numpy 数组
        sr: 采样率
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {file_path}")
    
    # 使用 librosa 加载并重采样
    audio, sr = librosa.load(str(file_path), sr=target_sr, mono=mono)
    
    # 确保是 float32 类型
    audio = audio.astype(np.float32)
    
    return audio, sr


def save_audio(
    audio: np.ndarray,
    file_path: Union[str, Path],
    sr: int = 16000
) -> None:
    """
    保存音频到文件
    
    Args:
        audio: 音频数据
        file_path: 保存路径
        sr: 采样率
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    sf.write(str(file_path), audio, sr)
    print(f"[Audio] 已保存: {file_path}")


def get_audio_duration(file_path: Union[str, Path]) -> float:
    """
    获取音频文件时长（秒）
    
    Args:
        file_path: 音频文件路径
        
    Returns:
        duration: 音频时长（秒）
    """
    file_path = Path(file_path)
    return librosa.get_duration(path=str(file_path))


def trim_audio(
    audio: np.ndarray, 
    sr: int,
    start_sec: float = 0,
    end_sec: Optional[float] = None
) -> np.ndarray:
    """
    裁剪音频
    
    Args:
        audio: 音频数据
        sr: 采样率
        start_sec: 开始时间（秒）
        end_sec: 结束时间（秒），None 表示到末尾
        
    Returns:
        裁剪后的音频
    """
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr) if end_sec else len(audio)
    
    return audio[start_sample:end_sample]


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    音频归一化到 [-1, 1] 范围
    
    Args:
        audio: 音频数据
        
    Returns:
        归一化后的音频
    """
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio


def add_noise(
    audio: np.ndarray, 
    noise_level: float = 0.01
) -> np.ndarray:
    """
    向音频添加高斯噪声
    
    Args:
        audio: 音频数据
        noise_level: 噪声级别
        
    Returns:
        添加噪声后的音频
    """
    noise = np.random.normal(0, noise_level, audio.shape)
    return audio + noise.astype(np.float32)


def generate_silence(duration_sec: float, sr: int = 16000) -> np.ndarray:
    """
    生成静音音频
    
    Args:
        duration_sec: 时长（秒）
        sr: 采样率
        
    Returns:
        静音音频数据
    """
    return np.zeros(int(duration_sec * sr), dtype=np.float32)


def generate_noise(
    duration_sec: float, 
    sr: int = 16000,
    noise_type: str = "white"
) -> np.ndarray:
    """
    生成噪声音频
    
    Args:
        duration_sec: 时长（秒）
        sr: 采样率
        noise_type: 噪声类型 ("white", "pink")
        
    Returns:
        噪声音频数据
    """
    n_samples = int(duration_sec * sr)
    
    if noise_type == "white":
        return np.random.randn(n_samples).astype(np.float32) * 0.1
    elif noise_type == "pink":
        # 简单的粉红噪声生成
        white = np.random.randn(n_samples)
        # 应用 1/f 滤波
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n_samples)
        freqs[0] = 1e-6  # 避免除零
        fft = fft / np.sqrt(freqs)
        pink = np.fft.irfft(fft, n_samples)
        return normalize_audio(pink).astype(np.float32) * 0.1
    else:
        raise ValueError(f"不支持的噪声类型: {noise_type}")


if __name__ == "__main__":
    # 测试函数
    print("生成测试音频...")
    
    # 生成 5 秒静音
    silence = generate_silence(5.0)
    print(f"静音: shape={silence.shape}, dtype={silence.dtype}")
    
    # 生成 5 秒白噪声
    white_noise = generate_noise(5.0, noise_type="white")
    print(f"白噪声: shape={white_noise.shape}, range=[{white_noise.min():.3f}, {white_noise.max():.3f}]")
    
    print("音频工具测试完成!")
