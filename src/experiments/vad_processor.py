"""
VAD (Voice Activity Detection) 预处理模块
用于检测音频中是否包含语音，作为幻觉缓解的前处理方案
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from typing import List, Tuple, Optional, Union
import warnings

from config import SAMPLE_RATE


class VADProcessor:
    """
    语音活动检测处理器
    支持多种 VAD 后端
    """
    
    def __init__(self, backend: str = "energy"):
        """
        初始化 VAD 处理器
        
        Args:
            backend: VAD 后端 ("energy", "webrtc", "silero")
        """
        self.backend = backend
        self._vad = None
        
        print(f"[VAD] 初始化 VAD 处理器: {backend}")
        
        if backend == "webrtc":
            self._init_webrtc()
        elif backend == "silero":
            self._init_silero()
        elif backend == "energy":
            pass  # 基于能量的简单方法，无需初始化
        else:
            raise ValueError(f"不支持的 VAD 后端: {backend}")
    
    def _init_webrtc(self):
        """初始化 WebRTC VAD"""
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad()
            self._vad.set_mode(3)  # 最激进的模式
            print("[VAD] WebRTC VAD 初始化成功")
        except ImportError:
            print("[VAD] WebRTC VAD 未安装，使用能量检测")
            self.backend = "energy"
    
    def _init_silero(self):
        """初始化 Silero VAD"""
        try:
            import torch
            # 加载 Silero VAD 模型
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False
            )
            self._vad = model
            self._get_speech_timestamps = utils[0]
            print("[VAD] Silero VAD 初始化成功")
        except Exception as e:
            print(f"[VAD] Silero VAD 初始化失败: {e}，使用能量检测")
            self.backend = "energy"
    
    def detect_speech(
        self,
        audio: np.ndarray,
        sr: int = SAMPLE_RATE,
        threshold: float = 0.02
    ) -> bool:
        """
        检测音频中是否包含语音
        
        Args:
            audio: 音频数据
            sr: 采样率
            threshold: 能量阈值
            
        Returns:
            是否检测到语音
        """
        if self.backend == "energy":
            return self._detect_by_energy(audio, threshold)
        elif self.backend == "webrtc":
            return self._detect_by_webrtc(audio, sr)
        elif self.backend == "silero":
            return self._detect_by_silero(audio, sr)
        else:
            return self._detect_by_energy(audio, threshold)
    
    def _detect_by_energy(
        self, 
        audio: np.ndarray, 
        threshold: float = 0.02
    ) -> bool:
        """
        基于能量的简单语音检测
        
        Args:
            audio: 音频数据
            threshold: RMS 能量阈值
            
        Returns:
            是否可能包含语音
        """
        # 计算 RMS 能量
        rms = np.sqrt(np.mean(audio ** 2))
        return rms > threshold
    
    def _detect_by_webrtc(
        self, 
        audio: np.ndarray, 
        sr: int = SAMPLE_RATE
    ) -> bool:
        """
        使用 WebRTC VAD 检测语音
        
        Args:
            audio: 音频数据
            sr: 采样率
            
        Returns:
            是否检测到语音
        """
        if self._vad is None:
            return self._detect_by_energy(audio)
        
        # WebRTC VAD 需要 16-bit PCM
        audio_int16 = (audio * 32768).astype(np.int16)
        
        # WebRTC VAD 支持 10, 20, 30 ms 的帧
        frame_duration_ms = 30
        frame_size = int(sr * frame_duration_ms / 1000)
        
        speech_frames = 0
        total_frames = 0
        
        for i in range(0, len(audio_int16) - frame_size, frame_size):
            frame = audio_int16[i:i + frame_size].tobytes()
            try:
                is_speech = self._vad.is_speech(frame, sr)
                if is_speech:
                    speech_frames += 1
                total_frames += 1
            except:
                pass
        
        # 如果超过 10% 的帧被检测为语音，则认为包含语音
        return (speech_frames / max(total_frames, 1)) > 0.1
    
    def _detect_by_silero(
        self, 
        audio: np.ndarray, 
        sr: int = SAMPLE_RATE
    ) -> bool:
        """
        使用 Silero VAD 检测语音
        
        Args:
            audio: 音频数据
            sr: 采样率
            
        Returns:
            是否检测到语音
        """
        if self._vad is None:
            return self._detect_by_energy(audio)
        
        import torch
        
        # 转换为 torch tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # 获取语音时间戳
        speech_timestamps = self._get_speech_timestamps(
            audio_tensor, 
            self._vad, 
            sampling_rate=sr
        )
        
        return len(speech_timestamps) > 0
    
    def get_speech_segments(
        self,
        audio: np.ndarray,
        sr: int = SAMPLE_RATE
    ) -> List[Tuple[int, int]]:
        """
        获取语音片段的起止位置
        
        Args:
            audio: 音频数据
            sr: 采样率
            
        Returns:
            语音片段列表 [(start, end), ...]
        """
        if self.backend == "silero" and self._vad is not None:
            import torch
            audio_tensor = torch.from_numpy(audio).float()
            timestamps = self._get_speech_timestamps(
                audio_tensor, 
                self._vad, 
                sampling_rate=sr
            )
            return [(t['start'], t['end']) for t in timestamps]
        
        # 对于其他后端，返回简单的基于能量的分割
        return self._get_segments_by_energy(audio, sr)
    
    def _get_segments_by_energy(
        self,
        audio: np.ndarray,
        sr: int = SAMPLE_RATE,
        frame_ms: int = 30,
        threshold: float = 0.02
    ) -> List[Tuple[int, int]]:
        """
        基于能量的语音分割
        """
        frame_size = int(sr * frame_ms / 1000)
        segments = []
        in_speech = False
        start = 0
        
        for i in range(0, len(audio) - frame_size, frame_size):
            frame = audio[i:i + frame_size]
            rms = np.sqrt(np.mean(frame ** 2))
            
            if rms > threshold:
                if not in_speech:
                    start = i
                    in_speech = True
            else:
                if in_speech:
                    segments.append((start, i))
                    in_speech = False
        
        if in_speech:
            segments.append((start, len(audio)))
        
        return segments
    
    def filter_audio(
        self,
        audio: np.ndarray,
        sr: int = SAMPLE_RATE
    ) -> Optional[np.ndarray]:
        """
        过滤非语音音频
        
        Args:
            audio: 输入音频
            sr: 采样率
            
        Returns:
            如果检测到语音返回音频，否则返回 None
        """
        if self.detect_speech(audio, sr):
            return audio
        return None
    
    def extract_speech(
        self,
        audio: np.ndarray,
        sr: int = SAMPLE_RATE,
        min_speech_ms: int = 100
    ) -> np.ndarray:
        """
        提取语音片段，去除静音部分
        
        Args:
            audio: 输入音频
            sr: 采样率
            min_speech_ms: 最小语音片段长度（毫秒）
            
        Returns:
            提取的语音音频
        """
        segments = self.get_speech_segments(audio, sr)
        
        if not segments:
            return audio  # 没有检测到语音，返回原始音频
        
        min_samples = int(sr * min_speech_ms / 1000)
        
        # 过滤太短的片段
        segments = [(s, e) for s, e in segments if e - s >= min_samples]
        
        if not segments:
            return audio
        
        # 拼接语音片段
        speech_audio = np.concatenate([audio[s:e] for s, e in segments])
        
        return speech_audio


def preprocess_with_vad(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    backend: str = "energy"
) -> Tuple[np.ndarray, bool]:
    """
    VAD 预处理函数
    
    Args:
        audio: 输入音频
        sr: 采样率
        backend: VAD 后端
        
    Returns:
        处理后的音频, 是否检测到语音
    """
    vad = VADProcessor(backend=backend)
    has_speech = vad.detect_speech(audio, sr)
    
    if has_speech:
        processed_audio = vad.extract_speech(audio, sr)
        return processed_audio, True
    else:
        return audio, False


if __name__ == "__main__":
    print("=" * 50)
    print("VAD 处理器测试")
    print("=" * 50)
    
    from utils.audio_utils import generate_silence, generate_noise
    
    # 创建 VAD 处理器
    vad = VADProcessor(backend="energy")
    
    # 测试静音
    silence = generate_silence(5.0)
    has_speech = vad.detect_speech(silence)
    print(f"\n静音测试: {'检测到语音' if has_speech else '无语音'}")
    
    # 测试噪声
    noise = generate_noise(5.0, noise_type="white")
    has_speech = vad.detect_speech(noise)
    print(f"白噪声测试: {'检测到语音' if has_speech else '无语音'}")
    
    # 测试高能量噪声
    loud_noise = noise * 10
    has_speech = vad.detect_speech(loud_noise)
    print(f"高能量噪声测试: {'检测到语音' if has_speech else '无语音'}")
    
    print("\nVAD 测试完成!")
