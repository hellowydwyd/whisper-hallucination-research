"""
ESC-50 数据集幻觉测试实验
========================

测试 Whisper 在真实环境声音上的幻觉现象

ESC-50 包含 50 类环境声音：
- 动物: 狗叫、猫叫、鸟鸣、蟋蟀等
- 自然: 雨声、海浪、雷声、风声等  
- 人类非语音: 咳嗽、脚步、笑声等
- 室内: 钟声、门铃、键盘敲击等
- 城市: 直升机、电锯、警笛等

运行方式:
    conda activate d2l
    python run_esc50_experiment.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt

from config import DATA_DIR, OUTPUT_DIR, NON_SPEECH_DIR
from models.whisper_model import WhisperASR
from utils.audio_utils import load_audio, get_audio_duration
from utils.metrics import detect_looping, check_boh_match
from experiments.boh_filter import BoHFilter, postprocess_transcription
from config import COMMON_HALLUCINATIONS


# ESC-50 类别映射
ESC50_CATEGORIES = {
    # 动物 (0-9)
    0: 'dog', 1: 'rooster', 2: 'pig', 3: 'cow', 4: 'frog',
    5: 'cat', 6: 'hen', 7: 'insects', 8: 'sheep', 9: 'crow',
    # 自然声音 (10-19)
    10: 'rain', 11: 'sea_waves', 12: 'crackling_fire', 13: 'crickets', 14: 'chirping_birds',
    15: 'water_drops', 16: 'wind', 17: 'pouring_water', 18: 'toilet_flush', 19: 'thunderstorm',
    # 人类非语音 (20-29)
    20: 'crying_baby', 21: 'sneezing', 22: 'clapping', 23: 'breathing', 24: 'coughing',
    25: 'footsteps', 26: 'laughing', 27: 'brushing_teeth', 28: 'snoring', 29: 'drinking_sipping',
    # 室内/家庭 (30-39)
    30: 'door_knock', 31: 'mouse_click', 32: 'keyboard_typing', 33: 'door_wood_creaks', 34: 'can_opening',
    35: 'washing_machine', 36: 'vacuum_cleaner', 37: 'clock_alarm', 38: 'clock_tick', 39: 'glass_breaking',
    # 城市/室外 (40-49)
    40: 'helicopter', 41: 'chainsaw', 42: 'siren', 43: 'car_horn', 44: 'engine',
    45: 'train', 46: 'church_bells', 47: 'airplane', 48: 'fireworks', 49: 'hand_saw',
}

ESC50_SUPER_CATEGORIES = {
    'animals': list(range(0, 10)),
    'natural': list(range(10, 20)),
    'human_non_speech': list(range(20, 30)),
    'interior': list(range(30, 40)),
    'exterior': list(range(40, 50)),
}


def get_esc50_metadata(audio_dir: Path) -> pd.DataFrame:
    """
    解析 ESC-50 文件名获取元数据
    
    文件名格式: {fold}-{clip_id}-{take}-{target}.wav
    例如: 1-100032-A-0.wav
    """
    audio_files = list(audio_dir.glob("*.wav"))
    
    metadata = []
    for f in audio_files:
        parts = f.stem.split('-')
        if len(parts) == 4:
            fold, clip_id, take, target = parts
            target = int(target)
            metadata.append({
                'file': f.name,
                'path': str(f),
                'fold': int(fold),
                'clip_id': clip_id,
                'take': take,
                'target': target,
                'category': ESC50_CATEGORIES.get(target, 'unknown'),
            })
    
    df = pd.DataFrame(metadata)
    
    # 添加超类别
    def get_super_category(target):
        for super_cat, targets in ESC50_SUPER_CATEGORIES.items():
            if target in targets:
                return super_cat
        return 'unknown'
    
    df['super_category'] = df['target'].apply(get_super_category)
    
    return df


def run_esc50_experiment(
    model_size: str = "large-v3",
    max_samples: int = None,
    sample_per_category: int = None
):
    """
    运行 ESC-50 数据集实验
    
    Args:
        model_size: Whisper 模型大小
        max_samples: 最大测试样本数，None 为全部
        sample_per_category: 每个类别测试的样本数，None 为全部
    """
    print("\n" + "=" * 60)
    print("   ESC-50 环境声音幻觉测试实验")
    print("=" * 60)
    
    # 查找 ESC-50 数据目录
    esc50_dir = NON_SPEECH_DIR / "esc50"
    audio_dir = esc50_dir / "audio"
    
    if not audio_dir.exists():
        # 尝试其他可能的路径
        possible_paths = [
            esc50_dir / "ESC-50-master" / "audio",
            esc50_dir / "audio",
            NON_SPEECH_DIR / "ESC-50-master" / "audio",
        ]
        for p in possible_paths:
            if p.exists():
                audio_dir = p
                break
    
    if not audio_dir.exists():
        print(f"[Error] 找不到 ESC-50 音频目录")
        print(f"[Error] 请先运行: python download_datasets.py --dataset esc50")
        return None
    
    print(f"[ESC-50] 音频目录: {audio_dir}")
    
    # 获取元数据
    metadata = get_esc50_metadata(audio_dir)
    print(f"[ESC-50] 总音频数: {len(metadata)}")
    print(f"[ESC-50] 类别数: {metadata['target'].nunique()}")
    
    # 采样
    if sample_per_category:
        # 每个类别采样指定数量
        sampled = metadata.groupby('target').apply(
            lambda x: x.sample(min(len(x), sample_per_category), random_state=42)
        ).reset_index(drop=True)
        metadata = sampled
        print(f"[ESC-50] 采样后: {len(metadata)} 个样本 (每类 {sample_per_category} 个)")
    
    if max_samples and len(metadata) > max_samples:
        metadata = metadata.sample(max_samples, random_state=42)
        print(f"[ESC-50] 限制样本数: {max_samples}")
    
    # 初始化模型
    print(f"\n[Model] 加载 Whisper {model_size}...")
    asr = WhisperASR(model_size=model_size)
    
    # 初始化 BoH 过滤器
    boh = BoHFilter()
    
    # 运行实验
    results = []
    
    print(f"\n[Experiment] 开始测试 {len(metadata)} 个音频...")
    
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="处理中"):
        try:
            # 转录
            result = asr.transcribe(row['path'])
            transcription = result['text']
            
            # 后处理
            processed = postprocess_transcription(transcription)
            
            # 分析
            record = {
                'file': row['file'],
                'category': row['category'],
                'super_category': row['super_category'],
                'target': row['target'],
                'transcription': transcription,
                'processed': processed,
                'is_hallucination': len(transcription.strip()) > 0,
                'is_looping': detect_looping(transcription),
                'boh_matches': check_boh_match(transcription, COMMON_HALLUCINATIONS),
                'char_count': len(transcription),
                'word_count': len(transcription.split()),
            }
            results.append(record)
            
        except Exception as e:
            print(f"\n[Error] {row['file']}: {e}")
    
    # 转换为 DataFrame
    df = pd.DataFrame(results)
    
    # 统计分析
    print("\n" + "=" * 60)
    print("实验结果统计")
    print("=" * 60)
    
    total = len(df)
    hallucination_count = df['is_hallucination'].sum()
    looping_count = df['is_looping'].sum()
    
    print(f"\n📊 总体统计:")
    print(f"  - 总样本数: {total}")
    print(f"  - 幻觉数: {hallucination_count}")
    print(f"  - 幻觉率: {hallucination_count/total:.1%}")
    print(f"  - 循环率: {looping_count/total:.1%}")
    
    # 按超类别统计
    print(f"\n📊 按声音类型统计:")
    super_stats = df.groupby('super_category').agg({
        'is_hallucination': ['sum', 'mean', 'count']
    }).round(3)
    super_stats.columns = ['幻觉数', '幻觉率', '样本数']
    print(super_stats.to_string())
    
    # 按具体类别统计
    print(f"\n📊 各类别幻觉率 (Top 10 最高):")
    category_stats = df.groupby('category')['is_hallucination'].agg(['sum', 'mean', 'count'])
    category_stats.columns = ['幻觉数', '幻觉率', '样本数']
    category_stats = category_stats.sort_values('幻觉率', ascending=False)
    print(category_stats.head(10).to_string())
    
    # 幻觉内容分析
    hallucinations = df[df['is_hallucination']]['transcription'].tolist()
    if hallucinations:
        print(f"\n📊 幻觉内容分析:")
        phrase_freq = Counter(hallucinations)
        print(f"  - 唯一幻觉数: {len(phrase_freq)}")
        print(f"  - Top 10 常见幻觉:")
        for phrase, count in phrase_freq.most_common(10):
            pct = count / len(hallucinations) * 100
            display = phrase[:50] + "..." if len(phrase) > 50 else phrase
            print(f"      '{display}' ({count}次, {pct:.1f}%)")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # CSV
    csv_path = OUTPUT_DIR / f"esc50_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n[Save] CSV: {csv_path}")
    
    # JSON 摘要
    summary = {
        'model': model_size,
        'dataset': 'ESC-50',
        'total_samples': total,
        'hallucination_count': int(hallucination_count),
        'hallucination_rate': float(hallucination_count / total),
        'looping_rate': float(looping_count / total),
        'by_super_category': {
            cat: {
                'count': int(stats['count']),
                'hallucination_rate': float(stats['mean'])
            }
            for cat, stats in df.groupby('super_category')['is_hallucination'].agg(['mean', 'count']).iterrows()
        },
        'top_hallucinations': phrase_freq.most_common(20) if hallucinations else [],
    }
    
    json_path = OUTPUT_DIR / f"esc50_summary_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Save] JSON: {json_path}")
    
    # 生成可视化
    generate_esc50_visualizations(df, OUTPUT_DIR, timestamp)
    
    return df, summary


def generate_esc50_visualizations(df: pd.DataFrame, output_dir: Path, timestamp: str):
    """
    生成 ESC-50 实验可视化
    """
    print("\n[可视化] 生成图表...")
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 图1: 按超类别的幻觉率
    ax1 = axes[0, 0]
    super_stats = df.groupby('super_category')['is_hallucination'].mean().sort_values(ascending=True)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(super_stats)))
    bars = ax1.barh(super_stats.index, super_stats.values, color=colors)
    ax1.set_xlabel('Hallucination Rate')
    ax1.set_title('Hallucination Rate by Sound Category', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 1)
    for bar, val in zip(bars, super_stats.values):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.1%}', va='center')
    
    # 图2: 按具体类别的幻觉率 (Top 15)
    ax2 = axes[0, 1]
    cat_stats = df.groupby('category')['is_hallucination'].mean().sort_values(ascending=False).head(15)
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(cat_stats)))
    bars = ax2.barh(range(len(cat_stats)), cat_stats.values, color=colors)
    ax2.set_yticks(range(len(cat_stats)))
    ax2.set_yticklabels(cat_stats.index)
    ax2.set_xlabel('Hallucination Rate')
    ax2.set_title('Top 15 Categories by Hallucination Rate', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.invert_yaxis()
    
    # 图3: 幻觉内容长度分布
    ax3 = axes[1, 0]
    hall_df = df[df['is_hallucination']]
    if len(hall_df) > 0:
        ax3.hist(hall_df['char_count'], bins=30, color='#e74c3c', edgecolor='white', alpha=0.8)
        ax3.axvline(hall_df['char_count'].mean(), color='black', linestyle='--', label=f"Mean: {hall_df['char_count'].mean():.1f}")
        ax3.set_xlabel('Character Count')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Hallucination Length', fontsize=12, fontweight='bold')
        ax3.legend()
    
    # 图4: 幻觉内容词云风格的频率图
    ax4 = axes[1, 1]
    if len(hall_df) > 0:
        phrase_freq = Counter(hall_df['transcription'].tolist())
        top_phrases = phrase_freq.most_common(10)
        if top_phrases:
            phrases, counts = zip(*top_phrases)
            phrases = [p[:30] + "..." if len(p) > 30 else p for p in phrases]
            y_pos = range(len(phrases))
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(phrases)))
            ax4.barh(y_pos, counts, color=colors)
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(phrases)
            ax4.set_xlabel('Frequency')
            ax4.set_title('Top 10 Hallucination Phrases', fontsize=12, fontweight='bold')
            ax4.invert_yaxis()
    
    plt.tight_layout()
    
    fig_path = output_dir / f"esc50_results_{timestamp}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"[可视化] 图表已保存: {fig_path}")
    
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ESC-50 数据集幻觉测试')
    parser.add_argument('--model', type=str, default='large-v3', help='Whisper 模型大小')
    parser.add_argument('--max-samples', type=int, default=None, help='最大测试样本数')
    parser.add_argument('--per-category', type=int, default=None, help='每类别测试样本数')
    parser.add_argument('--quick', action='store_true', help='快速测试模式 (每类5个)')
    
    args = parser.parse_args()
    
    if args.quick:
        args.per_category = 5
        print("[Mode] 快速测试模式: 每类别 5 个样本")
    
    df, summary = run_esc50_experiment(
        model_size=args.model,
        max_samples=args.max_samples,
        sample_per_category=args.per_category
    )
    
    if summary:
        print("\n" + "=" * 60)
        print("🎉 ESC-50 实验完成!")
        print("=" * 60)
        print(f"  模型: {summary['model']}")
        print(f"  样本数: {summary['total_samples']}")
        print(f"  幻觉率: {summary['hallucination_rate']:.1%}")


if __name__ == "__main__":
    main()
