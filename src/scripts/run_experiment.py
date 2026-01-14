"""
Whisper ASR 幻觉研究 - 主实验脚本
===============================

运行方式:
    python run_experiment.py --mode quick    # 快速测试
    python run_experiment.py --mode full     # 完整实验
    python run_experiment.py --mode custom   # 自定义实验

环境准备:
    conda activate d2l
    cd "D:\DeskTop\COURSES\sound_processing\final work\src"
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import argparse
import json
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from config import OUTPUT_DIR, DATA_DIR, NON_SPEECH_DIR
from experiments.hallucination_test import HallucinationExperiment
from experiments.vad_processor import VADProcessor, preprocess_with_vad
from experiments.boh_filter import BoHFilter, DeloopFilter, postprocess_transcription
from utils.audio_utils import generate_silence, generate_noise


def run_quick_test(model_size: str = "base"):
    """
    快速测试模式 - 验证代码正常工作
    """
    print("\n" + "=" * 60)
    print("🚀 快速测试模式")
    print("=" * 60)
    
    # 创建实验
    exp = HallucinationExperiment(model_size=model_size)
    
    # 测试静音
    print("\n[1/3] 测试静音...")
    silence_df = exp.test_silence(
        durations=[1, 5, 10],
        num_samples=2
    )
    
    # 测试噪声
    print("\n[2/3] 测试噪声...")
    noise_df = exp.test_noise(
        noise_types=["white"],
        durations=[1, 5, 10],
        num_samples=2
    )
    
    # 分析结果
    print("\n[3/3] 分析结果...")
    analysis = exp.analyze_hallucination_content()
    
    # 保存结果
    exp.save_results("quick_test_results.csv")
    
    return exp.get_summary()


def run_full_experiment(model_size: str = "base"):
    """
    完整实验模式 - 论文实验复现
    """
    print("\n" + "=" * 60)
    print("🔬 完整实验模式")
    print("=" * 60)
    
    results = {}
    
    # ===== 实验1: 不同类型非语音音频的幻觉测试 =====
    print("\n" + "-" * 40)
    print("实验1: 非语音音频幻觉率测试")
    print("-" * 40)
    
    exp = HallucinationExperiment(model_size=model_size)
    
    # 测试静音
    silence_df = exp.test_silence(
        durations=[1, 5, 10, 20, 30],
        num_samples=5
    )
    
    # 测试不同噪声
    noise_df = exp.test_noise(
        noise_types=["white", "pink"],
        durations=[1, 5, 10, 20, 30],
        num_samples=3
    )
    
    results['non_speech_test'] = exp.get_summary()
    
    # ===== 实验2: 音频长度对幻觉的影响 =====
    print("\n" + "-" * 40)
    print("实验2: 音频长度对幻觉的影响")
    print("-" * 40)
    
    length_results = []
    durations = [1, 2, 5, 10, 15, 20, 25, 30]
    
    for duration in durations:
        # 对每个长度测试多次
        for i in range(5):
            # 静音
            audio = generate_silence(duration)
            result = exp.model.transcribe(audio)
            length_results.append({
                'duration': duration,
                'type': 'silence',
                'transcription': result['text'],
                'is_hallucination': len(result['text'].strip()) > 0
            })
            
            # 白噪声
            audio = generate_noise(duration, noise_type="white")
            result = exp.model.transcribe(audio)
            length_results.append({
                'duration': duration,
                'type': 'white_noise',
                'transcription': result['text'],
                'is_hallucination': len(result['text'].strip()) > 0
            })
    
    length_df = pd.DataFrame(length_results)
    results['length_analysis'] = {
        'by_duration': length_df.groupby('duration')['is_hallucination'].mean().to_dict()
    }
    
    # ===== 实验3: Whisper 参数对幻觉的影响 =====
    print("\n" + "-" * 40)
    print("实验3: 参数敏感性分析")
    print("-" * 40)
    
    # 测试 no_speech_threshold 参数
    param_df = exp.test_parameter_sensitivity(
        param_name="no_speech_threshold",
        param_values=[0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    )
    
    results['param_sensitivity'] = param_df.to_dict('records')
    
    # ===== 实验4: 后处理方法对比 =====
    print("\n" + "-" * 40)
    print("实验4: 后处理方法对比")
    print("-" * 40)
    
    # 收集所有幻觉样本
    all_transcriptions = [r['transcription'] for r in exp.results if r.get('is_hallucination')]
    
    if all_transcriptions:
        boh = BoHFilter()
        deloop = DeloopFilter()
        
        postprocess_results = []
        
        for text in all_transcriptions[:50]:  # 取前50个样本
            original_len = len(text)
            
            # 只用去循环
            delooped = deloop.deloop(text)
            
            # 只用 BoH
            boh_filtered = boh.filter(text, remove_all=True)
            
            # 两者结合
            combined = postprocess_transcription(text, use_boh=True, use_deloop=True)
            
            postprocess_results.append({
                'original': text[:50],
                'original_len': original_len,
                'deloop_len': len(delooped),
                'boh_len': len(boh_filtered),
                'combined_len': len(combined),
            })
        
        postprocess_df = pd.DataFrame(postprocess_results)
        results['postprocess_comparison'] = {
            'avg_original_len': postprocess_df['original_len'].mean(),
            'avg_deloop_len': postprocess_df['deloop_len'].mean(),
            'avg_boh_len': postprocess_df['boh_len'].mean(),
            'avg_combined_len': postprocess_df['combined_len'].mean(),
        }
    
    # 保存所有结果
    exp.save_results("full_experiment_results.csv")
    
    # 保存汇总
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = OUTPUT_DIR / f"experiment_summary_{timestamp}.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n[结果] 汇总已保存: {summary_path}")
    
    # 生成可视化
    generate_visualizations(exp.results, OUTPUT_DIR)
    
    return results


def generate_visualizations(results: list, output_dir: Path):
    """
    生成实验结果可视化图表
    """
    print("\n[可视化] 生成图表...")
    
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        print("[Warning] 没有数据可视化")
        return
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 图1: 不同音频类型的幻觉率
    if 'audio_type' in df.columns:
        ax1 = axes[0, 0]
        type_stats = df.groupby('audio_type')['is_hallucination'].mean()
        type_stats.plot(kind='bar', ax=ax1, color=['#3498db', '#e74c3c', '#2ecc71'])
        ax1.set_title('Hallucination Rate by Audio Type', fontsize=12, pad=20)
        ax1.set_xlabel('Audio Type')
        ax1.set_ylabel('Hallucination Rate')
        ax1.set_ylim(0, 1.15)  # 增加上限给标签留空间
        ax1.tick_params(axis='x', rotation=45)
        
        # 添加数值标签（调整位置避免与标题重叠）
        for i, v in enumerate(type_stats):
            ax1.text(i, v + 0.03, f'{v:.1%}', ha='center', fontsize=9)
    
    # 图2: 不同音频长度的幻觉率
    if 'duration' in df.columns:
        ax2 = axes[0, 1]
        duration_stats = df.groupby('duration')['is_hallucination'].mean()
        duration_stats.plot(kind='line', marker='o', ax=ax2, color='#9b59b6', linewidth=2)
        ax2.set_title('Hallucination Rate by Audio Duration', fontsize=12)
        ax2.set_xlabel('Duration (seconds)')
        ax2.set_ylabel('Hallucination Rate')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
    
    # 图3: 幻觉内容长度分布
    if 'char_count' in df.columns:
        ax3 = axes[1, 0]
        hallucination_df = df[df['is_hallucination'] == True]
        if len(hallucination_df) > 0:
            ax3.hist(hallucination_df['char_count'], bins=20, color='#e67e22', edgecolor='white')
            ax3.set_title('Distribution of Hallucination Length', fontsize=12)
            ax3.set_xlabel('Character Count')
            ax3.set_ylabel('Frequency')
    
    # 图4: 循环检测统计
    if 'is_looping' in df.columns:
        ax4 = axes[1, 1]
        loop_stats = df.groupby('audio_type')[['is_hallucination', 'is_looping']].mean()
        loop_stats.plot(kind='bar', ax=ax4, color=['#3498db', '#e74c3c'])
        ax4.set_title('Hallucination vs Looping Rate', fontsize=12)
        ax4.set_xlabel('Audio Type')
        ax4.set_ylabel('Rate')
        ax4.legend(['Hallucination', 'Looping'])
        ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout(h_pad=3, w_pad=2)
    
    # 保存图表
    fig_path = output_dir / "experiment_results.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"[可视化] 图表已保存: {fig_path}")
    
    plt.close()


def run_custom_experiment():
    """
    自定义实验模式 - 交互式配置
    """
    print("\n" + "=" * 60)
    print("🛠️ 自定义实验模式")
    print("=" * 60)
    
    print("\n请选择要运行的实验:")
    print("  1. 静音测试")
    print("  2. 噪声测试")
    print("  3. 参数敏感性测试")
    print("  4. 后处理方法对比")
    print("  5. 全部运行")
    
    choice = input("\n请输入选项 (1-5): ").strip()
    
    model_size = input("请输入模型大小 (tiny/base/small/medium/large) [默认 base]: ").strip()
    if not model_size:
        model_size = "base"
    
    exp = HallucinationExperiment(model_size=model_size)
    
    if choice == "1" or choice == "5":
        exp.test_silence(durations=[1, 5, 10, 20, 30], num_samples=5)
    
    if choice == "2" or choice == "5":
        exp.test_noise(noise_types=["white", "pink"], durations=[1, 5, 10, 20, 30], num_samples=3)
    
    if choice == "3" or choice == "5":
        exp.test_parameter_sensitivity(param_name="no_speech_threshold")
    
    if choice == "4" or choice == "5":
        exp.analyze_hallucination_content()
    
    exp.save_results("custom_experiment_results.csv")
    
    return exp.get_summary()


def main():
    parser = argparse.ArgumentParser(description='Whisper ASR 幻觉研究实验')
    parser.add_argument(
        '--mode', 
        type=str, 
        default='quick',
        choices=['quick', 'full', 'custom'],
        help='实验模式: quick(快速测试), full(完整实验), custom(自定义)'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='base',
        help='Whisper 模型大小 (tiny, base, small, medium, large)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("   Whisper ASR 幻觉研究实验系统")
    print("=" * 60)
    print(f"模式: {args.mode}")
    print(f"模型: {args.model}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.mode == 'quick':
        summary = run_quick_test(model_size=args.model)
    elif args.mode == 'full':
        summary = run_full_experiment(model_size=args.model)
    else:
        summary = run_custom_experiment()
    
    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)
    
    if summary:
        print(f"\n📊 实验摘要:")
        # 处理不同返回格式
        if 'non_speech_test' in summary:
            # full experiment 返回格式
            ns = summary.get('non_speech_test', {})
            print(f"  - 模型: {ns.get('model_size', 'N/A')}")
            print(f"  - 总样本数: {ns.get('total_samples', 'N/A')}")
            print(f"  - 总体幻觉率: {ns.get('hallucination_rate', 0):.1%}")
            print(f"  - 总体循环率: {ns.get('looping_rate', 0):.1%}")
            if 'by_audio_type' in ns:
                print(f"  - 按类型:")
                for atype, stats in ns['by_audio_type'].items():
                    print(f"      {atype}: 幻觉率={stats['hallucination_rate']:.1%}")
        else:
            # get_summary() 返回格式
            print(f"  - 总样本数: {summary.get('total_samples', 'N/A')}")
            print(f"  - 总体幻觉率: {summary.get('hallucination_rate', 0):.1%}")
            print(f"  - 总体循环率: {summary.get('looping_rate', 0):.1%}")
    
    print(f"\n📁 结果保存位置: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
