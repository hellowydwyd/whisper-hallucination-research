"""
评估指标计算
"""
from typing import List, Dict, Any, Optional
import re


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    计算词错误率 (Word Error Rate)
    
    WER = (S + D + I) / N
    - S: 替换词数
    - D: 删除词数  
    - I: 插入词数
    - N: 参考文本总词数
    
    Args:
        reference: 参考文本（正确答案）
        hypothesis: 假设文本（模型输出）
        
    Returns:
        WER 值 (0.0 - 1.0+)
    """
    try:
        from jiwer import wer
        return wer(reference, hypothesis)
    except ImportError:
        # 简单实现
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        if len(ref_words) == 0:
            return 1.0 if len(hyp_words) > 0 else 0.0
        
        # 使用动态规划计算编辑距离
        d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
        
        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j
            
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(
                        d[i-1][j] + 1,      # 删除
                        d[i][j-1] + 1,      # 插入
                        d[i-1][j-1] + 1     # 替换
                    )
        
        return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    计算字符错误率 (Character Error Rate)
    
    Args:
        reference: 参考文本
        hypothesis: 假设文本
        
    Returns:
        CER 值
    """
    try:
        from jiwer import cer
        return cer(reference, hypothesis)
    except ImportError:
        ref_chars = list(reference.lower())
        hyp_chars = list(hypothesis.lower())
        
        if len(ref_chars) == 0:
            return 1.0 if len(hyp_chars) > 0 else 0.0
            
        # 编辑距离
        d = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]
        
        for i in range(len(ref_chars) + 1):
            d[i][0] = i
        for j in range(len(hyp_chars) + 1):
            d[0][j] = j
            
        for i in range(1, len(ref_chars) + 1):
            for j in range(1, len(hyp_chars) + 1):
                if ref_chars[i-1] == hyp_chars[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+1)
                    
        return d[len(ref_chars)][len(hyp_chars)] / len(ref_chars)


def calculate_hallucination_rate(
    results: List[Dict[str, Any]],
    empty_reference: bool = True
) -> float:
    """
    计算幻觉率
    
    幻觉率 = 产生幻觉的样本数 / 总样本数
    
    Args:
        results: 识别结果列表，每个元素包含 'transcription' 字段
        empty_reference: 是否期望输出为空（非语音音频测试）
        
    Returns:
        幻觉率 (0.0 - 1.0)
    """
    if len(results) == 0:
        return 0.0
    
    hallucination_count = 0
    
    for result in results:
        transcription = result.get('transcription', '').strip()
        
        if empty_reference:
            # 对于非语音音频，任何非空输出都是幻觉
            if len(transcription) > 0:
                hallucination_count += 1
        else:
            # 对于有参考的情况，需要比较
            reference = result.get('reference', '').strip()
            if transcription != reference:
                hallucination_count += 1
    
    return hallucination_count / len(results)


def detect_looping(text: str, min_repeat: int = 3) -> bool:
    """
    检测文本中是否存在循环重复
    
    Args:
        text: 输入文本
        min_repeat: 最小重复次数
        
    Returns:
        是否存在循环
    """
    if not text or len(text) < 10:
        return False
    
    words = text.lower().split()
    
    if len(words) < min_repeat:
        return False
    
    # 检测连续重复的词或短语
    for phrase_len in range(1, min(10, len(words) // min_repeat + 1)):
        for start in range(len(words) - phrase_len * min_repeat + 1):
            phrase = tuple(words[start:start + phrase_len])
            repeat_count = 1
            
            pos = start + phrase_len
            while pos + phrase_len <= len(words):
                if tuple(words[pos:pos + phrase_len]) == phrase:
                    repeat_count += 1
                    pos += phrase_len
                else:
                    break
                    
            if repeat_count >= min_repeat:
                return True
    
    return False


def calculate_looping_rate(results: List[Dict[str, Any]]) -> float:
    """
    计算循环率
    
    Args:
        results: 识别结果列表
        
    Returns:
        循环率
    """
    if len(results) == 0:
        return 0.0
    
    looping_count = sum(
        1 for r in results 
        if detect_looping(r.get('transcription', ''))
    )
    
    return looping_count / len(results)


def check_boh_match(
    text: str, 
    boh_phrases: List[str]
) -> List[str]:
    """
    检查文本是否包含 BoH (Bag of Hallucinations) 中的短语
    
    Args:
        text: 输入文本
        boh_phrases: 幻觉短语列表
        
    Returns:
        匹配到的幻觉短语列表
    """
    text_lower = text.lower()
    matches = []
    
    for phrase in boh_phrases:
        if phrase.lower() in text_lower:
            matches.append(phrase)
    
    return matches


if __name__ == "__main__":
    # 测试
    print("测试 WER 计算...")
    ref = "hello world how are you"
    hyp = "hello word how is you"
    wer_val = calculate_wer(ref, hyp)
    print(f"Reference: {ref}")
    print(f"Hypothesis: {hyp}")
    print(f"WER: {wer_val:.2%}")
    
    print("\n测试循环检测...")
    looping_text = "hello hello hello hello hello"
    print(f"Text: {looping_text}")
    print(f"Is looping: {detect_looping(looping_text)}")
    
    normal_text = "hello world how are you today"
    print(f"Text: {normal_text}")
    print(f"Is looping: {detect_looping(normal_text)}")
    
    print("\n指标测试完成!")
