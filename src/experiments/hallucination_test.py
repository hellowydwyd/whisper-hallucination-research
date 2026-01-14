"""
幻觉测试实验模块
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm
import json
from datetime import datetime

from config import (
    DATA_DIR, OUTPUT_DIR, NON_SPEECH_DIR, SPEECH_DIR,
    SAMPLE_RATE, COMMON_HALLUCINATIONS
)
from utils.audio_utils import (
    load_audio, generate_silence, generate_noise, 
    get_audio_duration, save_audio
)
from utils.metrics import (
    calculate_wer, calculate_hallucination_rate, 
    calculate_looping_rate, detect_looping, check_boh_match
)
from models.whisper_model import WhisperASR


class HallucinationExperiment:
    """
    Whisper 幻觉实验类
    """
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cuda",
        language: str = "en"
    ):
        """
        初始化实验
        
        Args:
            model_size: Whisper 模型大小
            device: 运行设备
            language: 识别语言
        """
        self.model_size = model_size
        self.device = device
        self.language = language
        
        # 延迟加载模型
        self._model = None
        
        # 实验结果存储
        self.results = []
        
    @property
    def model(self) -> WhisperASR:
        """懒加载模型"""
        if self._model is None:
            print(f"[Experiment] 初始化 Whisper {self.model_size} 模型...")
            self._model = WhisperASR(
                model_size=self.model_size,
                device=self.device,
                language=self.language
            )
        return self._model
    
    def test_silence(
        self, 
        durations: List[float] = [1, 5, 10, 20, 30],
        num_samples: int = 5
    ) -> pd.DataFrame:
        """
        测试静音音频的幻觉现象
        
        Args:
            durations: 测试的音频时长列表（秒）
            num_samples: 每个时长的样本数
            
        Returns:
            实验结果 DataFrame
        """
        print("\n" + "=" * 50)
        print("实验：静音音频幻觉测试")
        print("=" * 50)
        
        results = []
        
        for duration in tqdm(durations, desc="测试不同时长"):
            for i in range(num_samples):
                # 生成静音
                audio = generate_silence(duration)
                
                # 转录
                result = self.model.transcribe(audio)
                
                # 记录结果
                record = {
                    'audio_type': 'silence',
                    'duration': duration,
                    'sample_id': i,
                    'transcription': result['text'],
                    'is_hallucination': len(result['text'].strip()) > 0,
                    'is_looping': detect_looping(result['text']),
                    'boh_matches': check_boh_match(result['text'], COMMON_HALLUCINATIONS),
                    'char_count': len(result['text']),
                }
                results.append(record)
        
        df = pd.DataFrame(results)
        self.results.extend(results)
        
        # 打印统计
        print("\n--- 静音测试结果统计 ---")
        for duration in durations:
            subset = df[df['duration'] == duration]
            hall_rate = subset['is_hallucination'].mean()
            loop_rate = subset['is_looping'].mean()
            print(f"  {duration}s: 幻觉率={hall_rate:.1%}, 循环率={loop_rate:.1%}")
        
        return df
    
    def test_noise(
        self,
        noise_types: List[str] = ["white", "pink"],
        durations: List[float] = [1, 5, 10, 20, 30],
        num_samples: int = 3
    ) -> pd.DataFrame:
        """
        测试噪声音频的幻觉现象
        
        Args:
            noise_types: 噪声类型列表
            durations: 测试的音频时长列表（秒）
            num_samples: 每个配置的样本数
            
        Returns:
            实验结果 DataFrame
        """
        print("\n" + "=" * 50)
        print("实验：噪声音频幻觉测试")
        print("=" * 50)
        
        results = []
        
        total = len(noise_types) * len(durations) * num_samples
        pbar = tqdm(total=total, desc="噪声测试")
        
        for noise_type in noise_types:
            for duration in durations:
                for i in range(num_samples):
                    # 生成噪声
                    audio = generate_noise(duration, noise_type=noise_type)
                    
                    # 转录
                    result = self.model.transcribe(audio)
                    
                    # 记录结果
                    record = {
                        'audio_type': f'{noise_type}_noise',
                        'duration': duration,
                        'sample_id': i,
                        'transcription': result['text'],
                        'is_hallucination': len(result['text'].strip()) > 0,
                        'is_looping': detect_looping(result['text']),
                        'boh_matches': check_boh_match(result['text'], COMMON_HALLUCINATIONS),
                        'char_count': len(result['text']),
                    }
                    results.append(record)
                    pbar.update(1)
        
        pbar.close()
        df = pd.DataFrame(results)
        self.results.extend(results)
        
        # 打印统计
        print("\n--- 噪声测试结果统计 ---")
        for noise_type in noise_types:
            subset = df[df['audio_type'] == f'{noise_type}_noise']
            hall_rate = subset['is_hallucination'].mean()
            loop_rate = subset['is_looping'].mean()
            print(f"  {noise_type}: 幻觉率={hall_rate:.1%}, 循环率={loop_rate:.1%}")
        
        return df
    
    def test_audio_files(
        self,
        audio_dir: Union[str, Path],
        expected_empty: bool = True
    ) -> pd.DataFrame:
        """
        测试目录中的音频文件
        
        Args:
            audio_dir: 音频文件目录
            expected_empty: 是否期望输出为空（非语音音频）
            
        Returns:
            实验结果 DataFrame
        """
        audio_dir = Path(audio_dir)
        
        if not audio_dir.exists():
            print(f"[Warning] 目录不存在: {audio_dir}")
            return pd.DataFrame()
        
        # 查找音频文件
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(audio_dir.glob(f'*{ext}'))
            audio_files.extend(audio_dir.glob(f'*{ext.upper()}'))
        
        if len(audio_files) == 0:
            print(f"[Warning] 目录中没有找到音频文件: {audio_dir}")
            return pd.DataFrame()
        
        print(f"\n[Experiment] 找到 {len(audio_files)} 个音频文件")
        
        results = []
        
        for audio_file in tqdm(audio_files, desc="处理音频文件"):
            try:
                # 获取音频信息
                duration = get_audio_duration(audio_file)
                
                # 转录
                result = self.model.transcribe(str(audio_file))
                
                # 记录结果
                record = {
                    'file': audio_file.name,
                    'audio_type': 'file',
                    'duration': duration,
                    'transcription': result['text'],
                    'is_hallucination': len(result['text'].strip()) > 0 if expected_empty else None,
                    'is_looping': detect_looping(result['text']),
                    'boh_matches': check_boh_match(result['text'], COMMON_HALLUCINATIONS),
                    'char_count': len(result['text']),
                    'detected_language': result.get('language', 'unknown'),
                }
                results.append(record)
                
            except Exception as e:
                print(f"[Error] 处理 {audio_file.name} 失败: {e}")
        
        df = pd.DataFrame(results)
        self.results.extend(results)
        
        if expected_empty and len(df) > 0:
            hall_rate = df['is_hallucination'].mean()
            loop_rate = df['is_looping'].mean()
            print(f"\n--- 文件测试结果 ---")
            print(f"  总文件数: {len(df)}")
            print(f"  幻觉率: {hall_rate:.1%}")
            print(f"  循环率: {loop_rate:.1%}")
        
        return df
    
    def test_parameter_sensitivity(
        self,
        audio: Optional[np.ndarray] = None,
        param_name: str = "no_speech_threshold",
        param_values: List[float] = None
    ) -> pd.DataFrame:
        """
        测试 Whisper 参数对幻觉的影响
        
        Args:
            audio: 测试音频，默认为 10 秒静音
            param_name: 参数名称
            param_values: 参数值列表
            
        Returns:
            实验结果 DataFrame
        """
        if audio is None:
            audio = generate_silence(10.0)
        
        if param_values is None:
            if param_name == "no_speech_threshold":
                param_values = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
            elif param_name == "beam_size":
                param_values = [1, 3, 5, 10]
            elif param_name == "temperature":
                param_values = [0.0, 0.2, 0.5, 0.8, 1.0]
            else:
                param_values = [0.1, 0.5, 1.0]
        
        print(f"\n[Experiment] 测试参数敏感性: {param_name}")
        
        results = []
        
        for value in tqdm(param_values, desc=f"测试 {param_name}"):
            kwargs = {param_name: value}
            
            try:
                result = self.model.transcribe_with_options(audio, **kwargs)
                
                record = {
                    'param_name': param_name,
                    'param_value': value,
                    'transcription': result['text'],
                    'is_hallucination': len(result['text'].strip()) > 0,
                    'char_count': len(result['text']),
                }
                results.append(record)
                
            except Exception as e:
                print(f"[Error] {param_name}={value} 失败: {e}")
        
        df = pd.DataFrame(results)
        
        print(f"\n--- {param_name} 参数测试结果 ---")
        for _, row in df.iterrows():
            status = "幻觉" if row['is_hallucination'] else "正常"
            print(f"  {param_name}={row['param_value']}: {status}, 字符数={row['char_count']}")
        
        return df
    
    def analyze_hallucination_content(self) -> Dict[str, Any]:
        """
        分析幻觉内容的分布
        
        Returns:
            分析结果
        """
        if not self.results:
            print("[Warning] 没有实验结果可分析")
            return {}
        
        df = pd.DataFrame(self.results)
        hallucinations = df[df['is_hallucination'] == True]['transcription'].tolist()
        
        if not hallucinations:
            print("[Info] 没有检测到幻觉")
            return {}
        
        # 统计词频
        from collections import Counter
        all_words = []
        for text in hallucinations:
            all_words.extend(text.lower().split())
        
        word_freq = Counter(all_words)
        
        # 统计短语
        phrase_freq = Counter(hallucinations)
        
        analysis = {
            'total_hallucinations': len(hallucinations),
            'unique_hallucinations': len(set(hallucinations)),
            'top_words': word_freq.most_common(20),
            'top_phrases': phrase_freq.most_common(10),
            'avg_length': np.mean([len(h) for h in hallucinations]),
            'boh_match_rate': sum(
                1 for r in self.results 
                if r.get('boh_matches') and len(r['boh_matches']) > 0
            ) / len(hallucinations) if hallucinations else 0,
        }
        
        print("\n--- 幻觉内容分析 ---")
        print(f"  总幻觉数: {analysis['total_hallucinations']}")
        print(f"  唯一幻觉数: {analysis['unique_hallucinations']}")
        print(f"  平均长度: {analysis['avg_length']:.1f} 字符")
        print(f"  BoH 匹配率: {analysis['boh_match_rate']:.1%}")
        print(f"\n  Top 10 幻觉短语:")
        for phrase, count in analysis['top_phrases'][:10]:
            print(f"    - '{phrase[:50]}...' ({count}次)" if len(phrase) > 50 else f"    - '{phrase}' ({count}次)")
        
        return analysis
    
    def save_results(self, filename: str = None) -> Path:
        """
        保存实验结果
        
        Args:
            filename: 文件名，默认按时间戳生成
            
        Returns:
            保存的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_results_{timestamp}.csv"
        
        output_path = OUTPUT_DIR / filename
        
        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"[Experiment] 结果已保存: {output_path}")
        
        # 同时保存 JSON 格式
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        return output_path
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取实验摘要
        
        Returns:
            摘要信息
        """
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        
        summary = {
            'model_size': self.model_size,
            'total_samples': len(df),
            'hallucination_rate': df['is_hallucination'].mean() if 'is_hallucination' in df else 0,
            'looping_rate': df['is_looping'].mean() if 'is_looping' in df else 0,
            'by_audio_type': {},
        }
        
        if 'audio_type' in df.columns:
            for audio_type in df['audio_type'].unique():
                subset = df[df['audio_type'] == audio_type]
                summary['by_audio_type'][audio_type] = {
                    'count': len(subset),
                    'hallucination_rate': subset['is_hallucination'].mean(),
                    'looping_rate': subset['is_looping'].mean(),
                }
        
        return summary


if __name__ == "__main__":
    # 快速测试
    print("=" * 60)
    print("Whisper 幻觉实验 - 快速测试")
    print("=" * 60)
    
    # 创建实验实例
    exp = HallucinationExperiment(model_size="base")
    
    # 测试静音（减少样本数加快测试）
    silence_results = exp.test_silence(
        durations=[1, 5, 10],
        num_samples=2
    )
    
    # 测试白噪声
    noise_results = exp.test_noise(
        noise_types=["white"],
        durations=[1, 5, 10],
        num_samples=2
    )
    
    # 分析结果
    analysis = exp.analyze_hallucination_content()
    
    # 保存结果
    exp.save_results("quick_test_results.csv")
    
    # 打印摘要
    summary = exp.get_summary()
    print("\n" + "=" * 60)
    print("实验摘要:")
    print(f"  总样本数: {summary['total_samples']}")
    print(f"  总体幻觉率: {summary['hallucination_rate']:.1%}")
    print(f"  总体循环率: {summary['looping_rate']:.1%}")
    print("=" * 60)
