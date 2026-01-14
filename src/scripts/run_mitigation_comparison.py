"""
幻觉缓解方案对比实验
==================

对比不同缓解方案对 Whisper 幻觉的抑制效果：
1. 原始 Whisper (无处理)
2. VAD 预处理
3. BoH 后处理
4. VAD + BoH 组合

运行方式:
    conda activate d2l
    python run_mitigation_comparison.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt

from config import DATA_DIR, OUTPUT_DIR, NON_SPEECH_DIR
from models.whisper_model import WhisperASR
from utils.audio_utils import load_audio, generate_silence, generate_noise
from utils.metrics import calculate_wer
from experiments.vad_processor import VADProcessor, preprocess_with_vad
from experiments.boh_filter import BoHFilter, DeloopFilter, postprocess_transcription


def run_mitigation_comparison(model_size: str = "large-v3"):
    """
    运行缓解方案对比实验
    """
    print("\n" + "=" * 60)
    print("   幻觉缓解方案对比实验")
    print("=" * 60)
    
    # 初始化
    print(f"\n[Model] 加载 Whisper {model_size}...")
    asr = WhisperASR(model_size=model_size)
    
    vad = VADProcessor(backend="energy")
    boh = BoHFilter()
    deloop = DeloopFilter()
    
    results = []
    
    # ===== 实验1: 合成音频测试 =====
    print("\n" + "-" * 40)
    print("实验1: 合成音频测试")
    print("-" * 40)
    
    # 生成测试音频
    test_audios = []
    
    # 静音
    for duration in [1, 5, 10, 20, 30]:
        for i in range(3):
            audio = generate_silence(duration)
            test_audios.append({
                'audio': audio,
                'type': 'silence',
                'duration': duration,
                'id': f'silence_{duration}s_{i}'
            })
    
    # 白噪声
    for duration in [1, 5, 10, 20, 30]:
        for i in range(3):
            audio = generate_noise(duration, noise_type="white")
            test_audios.append({
                'audio': audio,
                'type': 'white_noise',
                'duration': duration,
                'id': f'white_noise_{duration}s_{i}'
            })
    
    # 粉红噪声
    for duration in [1, 5, 10, 20, 30]:
        for i in range(3):
            audio = generate_noise(duration, noise_type="pink")
            test_audios.append({
                'audio': audio,
                'type': 'pink_noise',
                'duration': duration,
                'id': f'pink_noise_{duration}s_{i}'
            })
    
    print(f"[Synthetic] 生成 {len(test_audios)} 个测试音频")
    
    # 测试每个音频
    for item in tqdm(test_audios, desc="合成音频测试"):
        audio = item['audio']
        
        # 方法1: 原始 Whisper
        result_raw = asr.transcribe(audio)
        text_raw = result_raw['text']
        
        # 方法2: VAD 预处理
        has_speech = vad.detect_speech(audio)
        if has_speech:
            result_vad = asr.transcribe(audio)
            text_vad = result_vad['text']
        else:
            text_vad = ""  # VAD 判定无语音，不转录
        
        # 方法3: BoH 后处理
        text_boh = postprocess_transcription(text_raw, use_boh=True, use_deloop=True)
        
        # 方法4: VAD + BoH 组合
        if has_speech:
            text_combined = postprocess_transcription(text_vad, use_boh=True, use_deloop=True)
        else:
            text_combined = ""
        
        # 记录结果
        record = {
            'id': item['id'],
            'type': item['type'],
            'duration': item['duration'],
            'source': 'synthetic',
            
            # 原始输出
            'raw_output': text_raw,
            'raw_is_hallucination': len(text_raw.strip()) > 0,
            'raw_char_count': len(text_raw),
            
            # VAD 预处理
            'vad_has_speech': has_speech,
            'vad_output': text_vad,
            'vad_is_hallucination': len(text_vad.strip()) > 0,
            'vad_char_count': len(text_vad),
            
            # BoH 后处理
            'boh_output': text_boh,
            'boh_is_hallucination': len(text_boh.strip()) > 0,
            'boh_char_count': len(text_boh),
            
            # VAD + BoH
            'combined_output': text_combined,
            'combined_is_hallucination': len(text_combined.strip()) > 0,
            'combined_char_count': len(text_combined),
        }
        results.append(record)
    
    # ===== 实验2: ESC-50 采样测试 =====
    print("\n" + "-" * 40)
    print("实验2: ESC-50 真实音频测试")
    print("-" * 40)
    
    esc50_audio_dir = NON_SPEECH_DIR / "esc50" / "audio"
    if not esc50_audio_dir.exists():
        esc50_audio_dir = NON_SPEECH_DIR / "esc50" / "ESC-50-master" / "audio"
    
    if esc50_audio_dir.exists():
        audio_files = list(esc50_audio_dir.glob("*.wav"))
        
        # 随机采样 100 个
        import random
        random.seed(42)
        sample_files = random.sample(audio_files, min(100, len(audio_files)))
        
        print(f"[ESC-50] 采样 {len(sample_files)} 个音频")
        
        for audio_file in tqdm(sample_files, desc="ESC-50 测试"):
            try:
                audio, sr = load_audio(audio_file)
                
                # 方法1: 原始 Whisper
                result_raw = asr.transcribe(audio)
                text_raw = result_raw['text']
                
                # 方法2: VAD 预处理
                has_speech = vad.detect_speech(audio)
                if has_speech:
                    text_vad = result_raw['text']  # 已经转录过了
                else:
                    text_vad = ""
                
                # 方法3: BoH 后处理
                text_boh = postprocess_transcription(text_raw, use_boh=True, use_deloop=True)
                
                # 方法4: VAD + BoH 组合
                text_combined = postprocess_transcription(text_vad, use_boh=True, use_deloop=True) if has_speech else ""
                
                record = {
                    'id': audio_file.stem,
                    'type': 'esc50',
                    'duration': len(audio) / sr,
                    'source': 'esc50',
                    
                    'raw_output': text_raw,
                    'raw_is_hallucination': len(text_raw.strip()) > 0,
                    'raw_char_count': len(text_raw),
                    
                    'vad_has_speech': has_speech,
                    'vad_output': text_vad,
                    'vad_is_hallucination': len(text_vad.strip()) > 0,
                    'vad_char_count': len(text_vad),
                    
                    'boh_output': text_boh,
                    'boh_is_hallucination': len(text_boh.strip()) > 0,
                    'boh_char_count': len(text_boh),
                    
                    'combined_output': text_combined,
                    'combined_is_hallucination': len(text_combined.strip()) > 0,
                    'combined_char_count': len(text_combined),
                }
                results.append(record)
                
            except Exception as e:
                print(f"\n[Error] {audio_file.name}: {e}")
    else:
        print("[Warning] ESC-50 数据不存在，跳过")
    
    # 分析结果
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print("缓解方案对比结果")
    print("=" * 60)
    
    # 总体统计
    print(f"\n📊 总体幻觉率对比:")
    print(f"  {'方法':<20} {'幻觉率':<12} {'平均输出长度':<15} {'降低幅度':<12}")
    print("-" * 60)
    
    raw_rate = df['raw_is_hallucination'].mean()
    vad_rate = df['vad_is_hallucination'].mean()
    boh_rate = df['boh_is_hallucination'].mean()
    combined_rate = df['combined_is_hallucination'].mean()
    
    raw_len = df['raw_char_count'].mean()
    vad_len = df['vad_char_count'].mean()
    boh_len = df['boh_char_count'].mean()
    combined_len = df['combined_char_count'].mean()
    
    print(f"  {'原始 Whisper':<20} {raw_rate:<12.1%} {raw_len:<15.1f} {'(基准)':<12}")
    print(f"  {'VAD 预处理':<20} {vad_rate:<12.1%} {vad_len:<15.1f} {(raw_rate-vad_rate)/raw_rate*100 if raw_rate > 0 else 0:>+.1f}%")
    print(f"  {'BoH 后处理':<20} {boh_rate:<12.1%} {boh_len:<15.1f} {(raw_rate-boh_rate)/raw_rate*100 if raw_rate > 0 else 0:>+.1f}%")
    print(f"  {'VAD + BoH 组合':<20} {combined_rate:<12.1%} {combined_len:<15.1f} {(raw_rate-combined_rate)/raw_rate*100 if raw_rate > 0 else 0:>+.1f}%")
    
    # 按数据源分类
    print(f"\n📊 按数据源分类:")
    for source in df['source'].unique():
        source_df = df[df['source'] == source]
        print(f"\n  [{source}] (n={len(source_df)})")
        print(f"    原始 Whisper: {source_df['raw_is_hallucination'].mean():.1%}")
        print(f"    VAD 预处理:   {source_df['vad_is_hallucination'].mean():.1%}")
        print(f"    BoH 后处理:   {source_df['boh_is_hallucination'].mean():.1%}")
        print(f"    VAD + BoH:    {source_df['combined_is_hallucination'].mean():.1%}")
    
    # VAD 效果分析
    print(f"\n📊 VAD 预处理分析:")
    vad_detected = df['vad_has_speech'].sum()
    print(f"  - 检测到语音的样本: {vad_detected} ({vad_detected/len(df):.1%})")
    print(f"  - 过滤掉的样本: {len(df) - vad_detected} ({(len(df)-vad_detected)/len(df):.1%})")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    csv_path = OUTPUT_DIR / f"mitigation_comparison_{timestamp}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n[Save] CSV: {csv_path}")
    
    summary = {
        'model': model_size,
        'total_samples': len(df),
        'methods': {
            'raw': {
                'hallucination_rate': float(raw_rate),
                'avg_char_count': float(raw_len),
            },
            'vad': {
                'hallucination_rate': float(vad_rate),
                'avg_char_count': float(vad_len),
                'reduction': float((raw_rate - vad_rate) / raw_rate) if raw_rate > 0 else 0,
            },
            'boh': {
                'hallucination_rate': float(boh_rate),
                'avg_char_count': float(boh_len),
                'reduction': float((raw_rate - boh_rate) / raw_rate) if raw_rate > 0 else 0,
            },
            'combined': {
                'hallucination_rate': float(combined_rate),
                'avg_char_count': float(combined_len),
                'reduction': float((raw_rate - combined_rate) / raw_rate) if raw_rate > 0 else 0,
            },
        }
    }
    
    json_path = OUTPUT_DIR / f"mitigation_comparison_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Save] JSON: {json_path}")
    
    # 生成可视化
    generate_comparison_visualization(df, summary, OUTPUT_DIR, timestamp)
    
    return df, summary


def generate_comparison_visualization(df: pd.DataFrame, summary: dict, output_dir: Path, timestamp: str):
    """
    生成缓解方案对比可视化
    """
    print("\n[可视化] 生成图表...")
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图1: 方法对比柱状图
    ax1 = axes[0, 0]
    methods = ['Raw\nWhisper', 'VAD\nPreprocess', 'BoH\nPostprocess', 'VAD +\nBoH']
    rates = [
        summary['methods']['raw']['hallucination_rate'],
        summary['methods']['vad']['hallucination_rate'],
        summary['methods']['boh']['hallucination_rate'],
        summary['methods']['combined']['hallucination_rate'],
    ]
    colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
    bars = ax1.bar(methods, rates, color=colors, edgecolor='white', linewidth=2)
    ax1.set_ylabel('Hallucination Rate')
    ax1.set_title('Hallucination Rate by Mitigation Method', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    for bar, rate in zip(bars, rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{rate:.1%}', ha='center', fontsize=11, fontweight='bold')
    
    # 图2: 按数据源分类对比
    ax2 = axes[0, 1]
    sources = df['source'].unique()
    x = np.arange(len(sources))
    width = 0.2
    
    raw_by_source = [df[df['source']==s]['raw_is_hallucination'].mean() for s in sources]
    vad_by_source = [df[df['source']==s]['vad_is_hallucination'].mean() for s in sources]
    boh_by_source = [df[df['source']==s]['boh_is_hallucination'].mean() for s in sources]
    combined_by_source = [df[df['source']==s]['combined_is_hallucination'].mean() for s in sources]
    
    ax2.bar(x - 1.5*width, raw_by_source, width, label='Raw', color='#e74c3c')
    ax2.bar(x - 0.5*width, vad_by_source, width, label='VAD', color='#f39c12')
    ax2.bar(x + 0.5*width, boh_by_source, width, label='BoH', color='#3498db')
    ax2.bar(x + 1.5*width, combined_by_source, width, label='Combined', color='#2ecc71')
    
    ax2.set_xlabel('Data Source')
    ax2.set_ylabel('Hallucination Rate')
    ax2.set_title('Hallucination Rate by Source', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sources)
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    
    # 图3: 输出长度对比
    ax3 = axes[1, 0]
    lengths = [
        summary['methods']['raw']['avg_char_count'],
        summary['methods']['vad']['avg_char_count'],
        summary['methods']['boh']['avg_char_count'],
        summary['methods']['combined']['avg_char_count'],
    ]
    bars = ax3.bar(methods, lengths, color=colors, edgecolor='white', linewidth=2)
    ax3.set_ylabel('Average Character Count')
    ax3.set_title('Average Output Length by Method', fontsize=12, fontweight='bold')
    for bar, length in zip(bars, lengths):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                f'{length:.1f}', ha='center', fontsize=10)
    
    # 图4: 降低幅度
    ax4 = axes[1, 1]
    reductions = [
        0,  # 基准
        summary['methods']['vad']['reduction'] * 100,
        summary['methods']['boh']['reduction'] * 100,
        summary['methods']['combined']['reduction'] * 100,
    ]
    colors_reduction = ['#95a5a6', '#f39c12', '#3498db', '#2ecc71']
    bars = ax4.bar(methods, reductions, color=colors_reduction, edgecolor='white', linewidth=2)
    ax4.set_ylabel('Reduction (%)')
    ax4.set_title('Hallucination Reduction Rate', fontsize=12, fontweight='bold')
    ax4.axhline(0, color='black', linewidth=0.5)
    for bar, red in zip(bars, reductions):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{red:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    
    fig_path = output_dir / f"mitigation_comparison_{timestamp}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"[可视化] 图表已保存: {fig_path}")
    
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='幻觉缓解方案对比实验')
    parser.add_argument('--model', type=str, default='large-v3', help='Whisper 模型大小')
    
    args = parser.parse_args()
    
    df, summary = run_mitigation_comparison(model_size=args.model)
    
    if summary:
        print("\n" + "=" * 60)
        print("🎉 缓解方案对比实验完成!")
        print("=" * 60)
        
        raw = summary['methods']['raw']['hallucination_rate']
        combined = summary['methods']['combined']['hallucination_rate']
        reduction = summary['methods']['combined']['reduction']
        
        print(f"  原始幻觉率: {raw:.1%}")
        print(f"  最优方案 (VAD+BoH): {combined:.1%}")
        print(f"  降低幅度: {reduction:.1%}")


if __name__ == "__main__":
    main()
