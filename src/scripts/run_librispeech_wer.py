"""
LibriSpeech WER 测试实验
========================

测试 Whisper 在正常语音上的词错误率 (WER)
验证模型在语音识别任务上的准确性

运行方式:
    conda activate d2l
    python run_librispeech_wer.py --quick     # 快速测试 (100 样本)
    python run_librispeech_wer.py             # 完整测试
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import re

from config import DATA_DIR, OUTPUT_DIR, SPEECH_DIR
from models.whisper_model import WhisperASR
from utils.metrics import calculate_wer, calculate_cer


def normalize_text(text: str) -> str:
    """
    标准化文本用于 WER 计算
    - 转小写
    - 移除标点
    - 规范化空格
    """
    text = text.lower()
    # 移除标点符号
    text = re.sub(r'[^\w\s]', '', text)
    # 规范化空格
    text = ' '.join(text.split())
    return text


def find_librispeech_data():
    """
    查找 LibriSpeech 数据目录和转录文件
    """
    possible_paths = [
        SPEECH_DIR / "librispeech" / "test-clean",
        SPEECH_DIR / "LibriSpeech" / "test-clean",
        DATA_DIR / "LibriSpeech" / "test-clean",
        DATA_DIR / "librispeech" / "test-clean",
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None


def load_librispeech_transcripts(data_dir: Path) -> dict:
    """
    加载 LibriSpeech 转录文本
    
    LibriSpeech 目录结构:
    test-clean/
        {speaker_id}/
            {chapter_id}/
                {speaker_id}-{chapter_id}-{utterance_id}.flac
                {speaker_id}-{chapter_id}.trans.txt
    """
    transcripts = {}
    
    # 遍历所有 trans.txt 文件
    for trans_file in data_dir.rglob("*.trans.txt"):
        with open(trans_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        utterance_id, text = parts
                        transcripts[utterance_id] = text
    
    return transcripts


def run_librispeech_wer(
    model_size: str = "large-v3",
    max_samples: int = None,
    quick: bool = False
):
    """
    运行 LibriSpeech WER 测试
    
    Args:
        model_size: Whisper 模型大小
        max_samples: 最大测试样本数
        quick: 快速测试模式
    """
    print("\n" + "=" * 60)
    print("   LibriSpeech WER 测试实验")
    print("=" * 60)
    
    # 查找数据目录
    data_dir = find_librispeech_data()
    
    if data_dir is None:
        print("[Error] 找不到 LibriSpeech 数据目录")
        print("[Error] 请先运行: python download_datasets.py --dataset librispeech")
        return None
    
    print(f"[LibriSpeech] 数据目录: {data_dir}")
    
    # 加载转录文本
    print("[LibriSpeech] 加载转录文本...")
    transcripts = load_librispeech_transcripts(data_dir)
    print(f"[LibriSpeech] 转录数量: {len(transcripts)}")
    
    # 查找音频文件
    audio_files = list(data_dir.rglob("*.flac"))
    print(f"[LibriSpeech] 音频文件数: {len(audio_files)}")
    
    if quick:
        max_samples = 100
        print(f"[LibriSpeech] 快速模式: 限制 {max_samples} 样本")
    
    if max_samples and len(audio_files) > max_samples:
        import random
        random.seed(42)
        audio_files = random.sample(audio_files, max_samples)
    
    # 初始化模型
    print(f"\n[Model] 加载 Whisper {model_size}...")
    asr = WhisperASR(model_size=model_size, language="en")
    
    # 运行测试
    results = []
    
    print(f"\n[Experiment] 开始测试 {len(audio_files)} 个音频...")
    
    for audio_file in tqdm(audio_files, desc="转录中"):
        try:
            # 获取 utterance ID
            utterance_id = audio_file.stem
            
            # 获取参考文本
            reference = transcripts.get(utterance_id, "")
            if not reference:
                continue
            
            # 转录
            result = asr.transcribe(str(audio_file))
            hypothesis = result['text']
            
            # 标准化
            ref_norm = normalize_text(reference)
            hyp_norm = normalize_text(hypothesis)
            
            # 计算 WER
            wer = calculate_wer(ref_norm, hyp_norm)
            cer = calculate_cer(ref_norm, hyp_norm)
            
            record = {
                'file': audio_file.name,
                'utterance_id': utterance_id,
                'reference': reference,
                'hypothesis': hypothesis,
                'ref_normalized': ref_norm,
                'hyp_normalized': hyp_norm,
                'wer': wer,
                'cer': cer,
                'ref_word_count': len(ref_norm.split()),
                'hyp_word_count': len(hyp_norm.split()),
            }
            results.append(record)
            
        except Exception as e:
            print(f"\n[Error] {audio_file.name}: {e}")
    
    # 分析结果
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print("WER 测试结果")
    print("=" * 60)
    
    # 计算总体 WER (加权平均)
    total_ref_words = df['ref_word_count'].sum()
    total_errors = (df['wer'] * df['ref_word_count']).sum()
    overall_wer = total_errors / total_ref_words if total_ref_words > 0 else 0
    
    # 计算总体 CER
    overall_cer = df['cer'].mean()
    
    print(f"\n📊 总体指标:")
    print(f"  - 测试样本数: {len(df)}")
    print(f"  - 总词数: {total_ref_words}")
    print(f"  - 总体 WER: {overall_wer:.2%}")
    print(f"  - 平均 CER: {overall_cer:.2%}")
    print(f"  - WER 标准差: {df['wer'].std():.2%}")
    
    # WER 分布统计
    print(f"\n📊 WER 分布:")
    print(f"  - 最小 WER: {df['wer'].min():.2%}")
    print(f"  - 25% 分位: {df['wer'].quantile(0.25):.2%}")
    print(f"  - 中位数:   {df['wer'].quantile(0.50):.2%}")
    print(f"  - 75% 分位: {df['wer'].quantile(0.75):.2%}")
    print(f"  - 最大 WER: {df['wer'].max():.2%}")
    
    # 完美识别比例
    perfect = (df['wer'] == 0).sum()
    print(f"\n📊 识别质量:")
    print(f"  - 完美识别 (WER=0): {perfect} ({perfect/len(df):.1%})")
    print(f"  - WER < 5%: {(df['wer'] < 0.05).sum()} ({(df['wer'] < 0.05).mean():.1%})")
    print(f"  - WER < 10%: {(df['wer'] < 0.10).sum()} ({(df['wer'] < 0.10).mean():.1%})")
    print(f"  - WER > 50%: {(df['wer'] > 0.50).sum()} ({(df['wer'] > 0.50).mean():.1%})")
    
    # 显示一些示例
    print(f"\n📊 识别示例 (前 5 个):")
    for i, row in df.head(5).iterrows():
        print(f"\n  [{i+1}] WER: {row['wer']:.1%}")
        print(f"      参考: {row['reference'][:60]}...")
        print(f"      识别: {row['hypothesis'][:60]}...")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    csv_path = OUTPUT_DIR / f"librispeech_wer_{timestamp}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n[Save] CSV: {csv_path}")
    
    summary = {
        'model': model_size,
        'dataset': 'LibriSpeech test-clean',
        'total_samples': len(df),
        'total_words': int(total_ref_words),
        'overall_wer': float(overall_wer),
        'overall_cer': float(overall_cer),
        'wer_std': float(df['wer'].std()),
        'wer_median': float(df['wer'].median()),
        'perfect_rate': float(perfect / len(df)),
    }
    
    json_path = OUTPUT_DIR / f"librispeech_wer_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Save] JSON: {json_path}")
    
    # 生成可视化
    generate_wer_visualization(df, OUTPUT_DIR, timestamp)
    
    return df, summary


def generate_wer_visualization(df: pd.DataFrame, output_dir: Path, timestamp: str):
    """
    生成 WER 可视化
    """
    import matplotlib.pyplot as plt
    
    print("\n[可视化] 生成图表...")
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图1: WER 分布直方图
    ax1 = axes[0, 0]
    ax1.hist(df['wer'], bins=50, color='#3498db', edgecolor='white', alpha=0.8)
    ax1.axvline(df['wer'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {df['wer'].mean():.2%}")
    ax1.axvline(df['wer'].median(), color='green', linestyle='--', linewidth=2, label=f"Median: {df['wer'].median():.2%}")
    ax1.set_xlabel('Word Error Rate (WER)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of WER', fontsize=12, fontweight='bold')
    ax1.legend()
    
    # 图2: WER 累积分布
    ax2 = axes[0, 1]
    sorted_wer = np.sort(df['wer'])
    cumulative = np.arange(1, len(sorted_wer) + 1) / len(sorted_wer)
    ax2.plot(sorted_wer, cumulative, color='#9b59b6', linewidth=2)
    ax2.axhline(0.9, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(sorted_wer[int(0.9 * len(sorted_wer))], color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Word Error Rate (WER)')
    ax2.set_ylabel('Cumulative Proportion')
    ax2.set_title('Cumulative Distribution of WER', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 图3: WER vs 句子长度
    ax3 = axes[1, 0]
    ax3.scatter(df['ref_word_count'], df['wer'], alpha=0.5, color='#e74c3c', s=20)
    ax3.set_xlabel('Reference Word Count')
    ax3.set_ylabel('WER')
    ax3.set_title('WER vs Sentence Length', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, min(df['wer'].max() * 1.1, 2.0))
    
    # 图4: WER 区间统计
    ax4 = axes[1, 1]
    bins = [0, 0.05, 0.10, 0.20, 0.50, 1.0, float('inf')]
    labels = ['0-5%', '5-10%', '10-20%', '20-50%', '50-100%', '>100%']
    df['wer_bin'] = pd.cut(df['wer'], bins=bins, labels=labels)
    bin_counts = df['wer_bin'].value_counts().reindex(labels)
    colors = ['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c', '#c0392b']
    ax4.bar(labels, bin_counts.values, color=colors, edgecolor='white')
    ax4.set_xlabel('WER Range')
    ax4.set_ylabel('Count')
    ax4.set_title('WER Distribution by Range', fontsize=12, fontweight='bold')
    for i, v in enumerate(bin_counts.values):
        ax4.text(i, v + 1, f'{v}', ha='center', fontsize=9)
    
    plt.tight_layout()
    
    fig_path = output_dir / f"librispeech_wer_{timestamp}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"[可视化] 图表已保存: {fig_path}")
    
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='LibriSpeech WER 测试')
    parser.add_argument('--model', type=str, default='large-v3', help='Whisper 模型大小')
    parser.add_argument('--max-samples', type=int, default=None, help='最大测试样本数')
    parser.add_argument('--quick', action='store_true', help='快速测试模式 (100 样本)')
    
    args = parser.parse_args()
    
    df, summary = run_librispeech_wer(
        model_size=args.model,
        max_samples=args.max_samples,
        quick=args.quick
    )
    
    if summary:
        print("\n" + "=" * 60)
        print("🎉 LibriSpeech WER 测试完成!")
        print("=" * 60)
        print(f"  模型: {summary['model']}")
        print(f"  样本数: {summary['total_samples']}")
        print(f"  总体 WER: {summary['overall_wer']:.2%}")


if __name__ == "__main__":
    main()
