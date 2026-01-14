[English](README.md) | **中文**

# Whisper ASR 幻觉问题研究与缓解方案

本项目针对 OpenAI Whisper 语音识别模型的幻觉（Hallucination）问题进行实验研究，并评估多种缓解方案的有效性。

## 项目简介

Whisper 模型在处理非语音音频（静音、噪声、环境声音）时，会生成与原始音频无关的文本输出，这种现象称为"幻觉"。本研究通过系统性实验验证了该问题，并对比了 VAD 预处理和 BoH 后处理等缓解方案。

## 目录结构

```
├── data/                    # 数据集
│   ├── non_speech/esc50/    # ESC-50 环境声音数据集
│   └── speech/librispeech/  # LibriSpeech 语音数据集
├── output/                  # 实验结果（CSV, PNG, JSON）
├── report/                  # 论文
│   ├── final_report.tex     # LaTeX 源文件
│   └── final_report.pdf     # 编译后 PDF
└── src/                     # 源代码
    ├── config.py            # 配置文件
    ├── requirements.txt     # 依赖列表
    ├── experiments/         # 实验模块
    ├── models/              # 模型封装
    ├── scripts/             # 运行脚本
    └── utils/               # 工具函数
```

## 环境配置

```bash
# 创建 conda 环境
conda create -n whisper python=3.9
conda activate whisper

# 安装依赖
pip install -r src/requirements.txt
```

## 运行实验

```bash
# 1. 下载数据集
python src/scripts/download_datasets.py

# 2. 合成音频幻觉测试
python src/scripts/run_experiment.py

# 3. ESC-50 环境声音测试
python src/scripts/run_esc50_experiment.py

# 4. LibriSpeech WER 测试
python src/scripts/run_librispeech_wer.py

# 5. 缓解方案对比
python src/scripts/run_mitigation_comparison.py
```

## 主要结论

| 实验 | 结果 |
|------|------|
| 合成音频幻觉率 | 100% |
| ESC-50 幻觉率 | 62.5% |
| LibriSpeech WER | 3.71% |
| BoH 缓解效果 | 幻觉率降低 74.1% |

## 参考文献

1. Radford, A., et al. Robust Speech Recognition via Large-Scale Weak Supervision. ICML, 2022.
2. Wang, Y., et al. Calm-Whisper: Reduce Whisper Hallucination. Interspeech, 2025.
3. Barański, M., et al. Investigation of Whisper ASR Hallucinations. ICASSP, 2025.
