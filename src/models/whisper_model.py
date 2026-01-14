"""
Whisper 模型封装
"""
import whisper
import torch
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, Optional, List
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_LANGUAGE, SAMPLE_RATE


class WhisperASR:
    """
    Whisper ASR 模型封装类
    """
    
    def __init__(
        self,
        model_size: str = WHISPER_MODEL_SIZE,
        device: str = WHISPER_DEVICE,
        language: Optional[str] = WHISPER_LANGUAGE
    ):
        """
        初始化 Whisper 模型
        
        Args:
            model_size: 模型大小 (tiny, base, small, medium, large, large-v2, large-v3)
            device: 运行设备 (cuda, cpu)
            language: 识别语言，None 为自动检测
        """
        self.model_size = model_size
        self.device = device if torch.cuda.is_available() else "cpu"
        self.language = language
        
        print(f"[Whisper] 加载模型: {model_size}")
        print(f"[Whisper] 设备: {self.device}")
        
        self.model = whisper.load_model(model_size, device=self.device)
        
        print(f"[Whisper] 模型加载完成!")
        
    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray],
        **kwargs
    ) -> Dict[str, Any]:
        """
        转录音频
        
        Args:
            audio: 音频文件路径或 numpy 数组
            **kwargs: 传递给 whisper.transcribe 的其他参数
            
        Returns:
            转录结果字典，包含:
            - text: 识别文本
            - segments: 分段信息
            - language: 检测到的语言
        """
        # 设置默认参数
        options = {
            'language': self.language,
            'task': 'transcribe',
            'verbose': False,
        }
        options.update(kwargs)
        
        # 如果是文件路径，转换为字符串
        if isinstance(audio, (str, Path)):
            audio = str(audio)
        
        result = self.model.transcribe(audio, **options)
        
        return {
            'text': result['text'].strip(),
            'segments': result['segments'],
            'language': result['language'],
            'full_result': result
        }
    
    def transcribe_batch(
        self,
        audio_files: List[Union[str, Path]],
        show_progress: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        批量转录音频文件
        
        Args:
            audio_files: 音频文件路径列表
            show_progress: 是否显示进度条
            **kwargs: 传递给 transcribe 的其他参数
            
        Returns:
            转录结果列表
        """
        results = []
        
        iterator = tqdm(audio_files, desc="转录中") if show_progress else audio_files
        
        for audio_file in iterator:
            try:
                result = self.transcribe(audio_file, **kwargs)
                result['file'] = str(audio_file)
                result['success'] = True
            except Exception as e:
                result = {
                    'file': str(audio_file),
                    'text': '',
                    'error': str(e),
                    'success': False
                }
            results.append(result)
        
        return results
    
    def transcribe_with_options(
        self,
        audio: Union[str, Path, np.ndarray],
        beam_size: int = 5,
        best_of: int = 5,
        temperature: float = 0.0,
        no_speech_threshold: float = 0.6,
        logprob_threshold: float = -1.0,
        compression_ratio_threshold: float = 2.4,
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用自定义参数转录（用于实验不同参数对幻觉的影响）
        
        Args:
            audio: 音频输入
            beam_size: 束搜索宽度
            best_of: 采样数量
            temperature: 采样温度
            no_speech_threshold: 无语音阈值
            logprob_threshold: 对数概率阈值
            compression_ratio_threshold: 压缩比阈值
            
        Returns:
            转录结果
        """
        return self.transcribe(
            audio,
            beam_size=beam_size,
            best_of=best_of,
            temperature=temperature,
            no_speech_threshold=no_speech_threshold,
            logprob_threshold=logprob_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            **kwargs
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            'model_size': self.model_size,
            'device': self.device,
            'language': self.language,
            'dims': {
                'n_mels': self.model.dims.n_mels,
                'n_audio_ctx': self.model.dims.n_audio_ctx,
                'n_audio_state': self.model.dims.n_audio_state,
                'n_audio_head': self.model.dims.n_audio_head,
                'n_audio_layer': self.model.dims.n_audio_layer,
                'n_vocab': self.model.dims.n_vocab,
                'n_text_ctx': self.model.dims.n_text_ctx,
                'n_text_state': self.model.dims.n_text_state,
                'n_text_head': self.model.dims.n_text_head,
                'n_text_layer': self.model.dims.n_text_layer,
            }
        }


def get_available_models() -> List[str]:
    """
    获取可用的 Whisper 模型列表
    """
    return ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]


if __name__ == "__main__":
    print("=" * 50)
    print("Whisper 模型测试")
    print("=" * 50)
    
    # 创建模型实例
    asr = WhisperASR(model_size="base")
    
    # 打印模型信息
    info = asr.get_model_info()
    print("\n模型信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\nWhisper 模型初始化成功!")
