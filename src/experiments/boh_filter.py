"""
BoH (Bag of Hallucinations) 后处理过滤模块
用于检测和移除常见的幻觉短语
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import re
from typing import List, Dict, Tuple, Optional
from collections import Counter

from config import COMMON_HALLUCINATIONS


class BoHFilter:
    """
    幻觉词袋过滤器
    
    基于 ICASSP 2025 论文中的方法，使用预定义的幻觉短语列表
    检测并移除 Whisper 输出中的常见幻觉内容
    """
    
    def __init__(self, hallucination_phrases: List[str] = None):
        """
        初始化 BoH 过滤器
        
        Args:
            hallucination_phrases: 幻觉短语列表，默认使用预定义列表
        """
        self.phrases = hallucination_phrases or COMMON_HALLUCINATIONS
        
        # 按长度降序排列，优先匹配更长的短语
        self.phrases = sorted(self.phrases, key=len, reverse=True)
        
        # 尝试使用 Aho-Corasick 算法加速匹配
        self._automaton = None
        self._init_automaton()
        
        print(f"[BoH] 初始化完成，包含 {len(self.phrases)} 个幻觉短语")
    
    def _init_automaton(self):
        """初始化 Aho-Corasick 自动机"""
        try:
            import ahocorasick
            self._automaton = ahocorasick.Automaton()
            
            for idx, phrase in enumerate(self.phrases):
                self._automaton.add_word(phrase.lower(), (idx, phrase))
            
            self._automaton.make_automaton()
            print("[BoH] Aho-Corasick 自动机初始化成功")
        except ImportError:
            print("[BoH] pyahocorasick 未安装，使用简单字符串匹配")
            self._automaton = None
    
    def detect(self, text: str) -> List[Dict]:
        """
        检测文本中的幻觉短语
        
        Args:
            text: 输入文本
            
        Returns:
            检测到的幻觉列表，每个元素包含:
            - phrase: 匹配的短语
            - start: 起始位置
            - end: 结束位置
        """
        if not text:
            return []
        
        text_lower = text.lower()
        matches = []
        
        if self._automaton is not None:
            # 使用 Aho-Corasick 算法
            for end_idx, (phrase_idx, phrase) in self._automaton.iter(text_lower):
                start_idx = end_idx - len(phrase) + 1
                matches.append({
                    'phrase': phrase,
                    'start': start_idx,
                    'end': end_idx + 1,
                })
        else:
            # 简单字符串匹配
            for phrase in self.phrases:
                phrase_lower = phrase.lower()
                start = 0
                while True:
                    idx = text_lower.find(phrase_lower, start)
                    if idx == -1:
                        break
                    matches.append({
                        'phrase': phrase,
                        'start': idx,
                        'end': idx + len(phrase),
                    })
                    start = idx + 1
        
        # 按起始位置排序
        matches.sort(key=lambda x: x['start'])
        
        return matches
    
    def filter(self, text: str, remove_all: bool = False) -> str:
        """
        过滤文本中的幻觉短语
        
        Args:
            text: 输入文本
            remove_all: 是否移除所有匹配，还是只移除完全匹配
            
        Returns:
            过滤后的文本
        """
        if not text:
            return text
        
        # 检测匹配
        matches = self.detect(text)
        
        if not matches:
            return text
        
        # 如果整个文本就是幻觉短语，返回空字符串
        text_stripped = text.strip().lower()
        for phrase in self.phrases:
            if text_stripped == phrase.lower():
                return ""
        
        if remove_all:
            # 移除所有匹配的幻觉短语
            # 从后向前移除，避免位置偏移问题
            result = text
            for match in reversed(matches):
                result = result[:match['start']] + result[match['end']:]
            
            # 清理多余空格
            result = re.sub(r'\s+', ' ', result).strip()
            return result
        else:
            return text
    
    def is_hallucination(self, text: str, threshold: float = 0.8) -> bool:
        """
        判断文本是否主要是幻觉
        
        Args:
            text: 输入文本
            threshold: 幻觉内容占比阈值
            
        Returns:
            是否是幻觉
        """
        if not text or not text.strip():
            return False
        
        matches = self.detect(text)
        
        if not matches:
            return False
        
        # 计算幻觉内容覆盖的字符数
        covered_chars = set()
        for match in matches:
            for i in range(match['start'], match['end']):
                covered_chars.add(i)
        
        # 计算覆盖比例
        total_chars = len(text.strip())
        coverage = len(covered_chars) / total_chars if total_chars > 0 else 0
        
        return coverage >= threshold
    
    def get_statistics(self, texts: List[str]) -> Dict:
        """
        统计一批文本中的幻觉情况
        
        Args:
            texts: 文本列表
            
        Returns:
            统计信息
        """
        all_matches = []
        hallucination_count = 0
        
        for text in texts:
            matches = self.detect(text)
            all_matches.extend([m['phrase'] for m in matches])
            
            if self.is_hallucination(text):
                hallucination_count += 1
        
        phrase_counter = Counter(all_matches)
        
        return {
            'total_texts': len(texts),
            'hallucination_count': hallucination_count,
            'hallucination_rate': hallucination_count / len(texts) if texts else 0,
            'total_matches': len(all_matches),
            'phrase_frequency': phrase_counter.most_common(),
        }
    
    def add_phrase(self, phrase: str):
        """添加新的幻觉短语"""
        if phrase.lower() not in [p.lower() for p in self.phrases]:
            self.phrases.append(phrase)
            self.phrases = sorted(self.phrases, key=len, reverse=True)
            self._init_automaton()  # 重新初始化自动机
    
    def remove_phrase(self, phrase: str):
        """移除幻觉短语"""
        self.phrases = [p for p in self.phrases if p.lower() != phrase.lower()]
        self._init_automaton()


class DeloopFilter:
    """
    循环检测和移除过滤器
    用于处理 Whisper 产生的循环幻觉（重复相同内容）
    """
    
    def __init__(self, min_repeat: int = 3):
        """
        初始化去循环过滤器
        
        Args:
            min_repeat: 最小重复次数阈值
        """
        self.min_repeat = min_repeat
    
    def detect_loops(self, text: str) -> List[Dict]:
        """
        检测文本中的循环模式
        
        Args:
            text: 输入文本
            
        Returns:
            检测到的循环列表
        """
        if not text:
            return []
        
        words = text.split()
        loops = []
        
        if len(words) < self.min_repeat:
            return loops
        
        # 检测不同长度的重复短语
        for phrase_len in range(1, min(10, len(words) // self.min_repeat + 1)):
            for start in range(len(words) - phrase_len * self.min_repeat + 1):
                phrase = tuple(words[start:start + phrase_len])
                
                # 计算连续重复次数
                repeat_count = 1
                pos = start + phrase_len
                
                while pos + phrase_len <= len(words):
                    if tuple(words[pos:pos + phrase_len]) == phrase:
                        repeat_count += 1
                        pos += phrase_len
                    else:
                        break
                
                if repeat_count >= self.min_repeat:
                    loops.append({
                        'phrase': ' '.join(phrase),
                        'repeat_count': repeat_count,
                        'start_word': start,
                        'end_word': start + phrase_len * repeat_count,
                    })
        
        return loops
    
    def deloop(self, text: str) -> str:
        """
        移除文本中的循环部分
        
        Args:
            text: 输入文本
            
        Returns:
            去循环后的文本
        """
        loops = self.detect_loops(text)
        
        if not loops:
            return text
        
        words = text.split()
        
        # 标记需要保留的词
        keep = [True] * len(words)
        
        for loop in loops:
            phrase_len = len(loop['phrase'].split())
            # 只保留第一次出现
            for i in range(loop['start_word'] + phrase_len, loop['end_word']):
                keep[i] = False
        
        # 重建文本
        result_words = [w for i, w in enumerate(words) if keep[i]]
        
        return ' '.join(result_words)
    
    def has_loop(self, text: str) -> bool:
        """判断文本是否包含循环"""
        return len(self.detect_loops(text)) > 0


def postprocess_transcription(
    text: str,
    use_boh: bool = True,
    use_deloop: bool = True
) -> str:
    """
    后处理转录结果
    
    Args:
        text: 原始转录文本
        use_boh: 是否使用 BoH 过滤
        use_deloop: 是否使用去循环
        
    Returns:
        处理后的文本
    """
    result = text
    
    if use_deloop:
        deloop = DeloopFilter()
        result = deloop.deloop(result)
    
    if use_boh:
        boh = BoHFilter()
        result = boh.filter(result, remove_all=True)
    
    return result.strip()


if __name__ == "__main__":
    print("=" * 50)
    print("BoH 过滤器测试")
    print("=" * 50)
    
    # 创建过滤器
    boh = BoHFilter()
    deloop = DeloopFilter()
    
    # 测试 BoH 检测
    test_texts = [
        "Thank you for watching!",
        "Hello world, how are you?",
        "Please subscribe and like this video",
        "thanks for watching please subscribe",
        "The quick brown fox",
    ]
    
    print("\n[BoH 检测测试]")
    for text in test_texts:
        matches = boh.detect(text)
        is_hall = boh.is_hallucination(text)
        print(f"  '{text[:40]}...' -> 幻觉: {is_hall}, 匹配: {[m['phrase'] for m in matches]}")
    
    # 测试去循环
    loop_texts = [
        "hello hello hello hello hello",
        "thank you thank you thank you thank you",
        "this is normal text without loops",
        "yes yes yes no no no yes yes yes",
    ]
    
    print("\n[去循环测试]")
    for text in loop_texts:
        has_loop = deloop.has_loop(text)
        cleaned = deloop.deloop(text)
        print(f"  '{text[:40]}...'")
        print(f"    -> 有循环: {has_loop}")
        print(f"    -> 清理后: '{cleaned}'")
    
    # 测试统计
    print("\n[统计测试]")
    stats = boh.get_statistics(test_texts)
    print(f"  总文本数: {stats['total_texts']}")
    print(f"  幻觉数: {stats['hallucination_count']}")
    print(f"  幻觉率: {stats['hallucination_rate']:.1%}")
    
    print("\nBoH 测试完成!")
