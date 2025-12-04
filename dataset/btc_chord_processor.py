#!/usr/bin/env python3
"""
BTC和弦识别处理器

使用Park 2019的BTC-ISMIR19模型进行和弦识别，并处理成13种和弦类型格式。
"""

import os
import sys
import subprocess
import tempfile
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from music21 import key, converter, pitch
except ImportError:
    music21_available = False
else:
    music21_available = True


def run_btc_inference(wav_path: Path, output_dir: Path, video_id: str, 
                     btc_python: str, btc_repo_path: Path) -> Tuple[bool, Optional[Path]]:
    """
    调用BTC模型进行推理
    
    Args:
        wav_path: WAV文件路径
        output_dir: 输出目录
        video_id: 视频ID
        btc_python: BTC环境的Python路径
        btc_repo_path: BTC仓库路径
    
    Returns:
        (成功标志, lab文件路径)
    """
    try:
        # 创建临时目录用于BTC输出
        temp_output = tempfile.mkdtemp(prefix='btc_')
        
        # 调用BTC模型的test.py脚本
        btc_test_script = btc_repo_path / 'test.py'
        if not btc_test_script.exists():
            return False, None
        
        # 创建临时音频目录（BTC需要目录）
        temp_audio_dir = Path(temp_output) / 'audio'
        temp_audio_dir.mkdir(parents=True)
        # 创建符号链接或复制文件
        import shutil
        temp_audio_file = temp_audio_dir / f'{video_id}.wav'
        shutil.copy2(wav_path, temp_audio_file)
        
        # BTC的test.py需要在仓库目录运行（需要读取run_config.yaml）
        result = subprocess.run(
            [btc_python, str(btc_test_script),
             '--audio_dir', str(temp_audio_dir),
             '--save_dir', str(temp_output),
             '--voca', 'True'],  # 使用大词表
            cwd=str(btc_repo_path),  # 在BTC仓库目录中运行
            capture_output=True,
            text=True,
            check=False,
            timeout=300  # 5分钟超时
        )
        
        if result.returncode != 0:
            print(f"BTC推理失败: {result.stderr}")
            return False, None
        
        # 查找生成的lab文件
        lab_files = list(Path(temp_output).glob('*.lab'))
        if not lab_files:
            # 也可能在子目录中
            lab_files = list(Path(temp_output).rglob('*.lab'))
        
        if lab_files:
            # 复制到目标目录
            output_dir.mkdir(parents=True, exist_ok=True)
            lab_path = output_dir / f'{video_id}.lab'
            shutil.copy2(lab_files[0], lab_path)
            # 清理临时目录
            shutil.rmtree(temp_output, ignore_errors=True)
            return True, lab_path
        else:
            shutil.rmtree(temp_output, ignore_errors=True)
            return False, None
            
    except Exception as e:
        print(f"BTC推理异常: {e}")
        return False, None


def parse_lab_file(lab_path: Path) -> List[Tuple[float, float, str]]:
    """
    解析BTC输出的lab文件
    
    Args:
        lab_path: lab文件路径
    
    Returns:
        [(start_time, end_time, chord_label), ...]
    """
    events = []
    try:
        with open(lab_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        start = float(parts[0])
                        end = float(parts[1])
                        chord = parts[2]
                        events.append((start, end, chord))
                    except ValueError:
                        continue
    except Exception as e:
        print(f"解析lab文件失败: {e}")
    
    return events


def resample_to_1hz(lab_events: List[Tuple[float, float, str]], 
                    total_duration: float) -> List[str]:
    """
    将BTC输出的lab事件重采样到每秒一个和弦
    
    Args:
        lab_events: [(start, end, chord), ...]
        total_duration: 总时长（秒）
    
    Returns:
        每秒一个和弦的列表
    """
    T = int(math.ceil(total_duration))
    result = []
    
    for sec in range(T):
        s, e = sec, sec + 1.0
        dur_by_chord = {}
        
        for st, et, lab in lab_events:
            # 计算交集
            inter = max(0.0, min(et, e) - max(st, s))
            if inter > 0:
                dur_by_chord[lab] = dur_by_chord.get(lab, 0.0) + inter
        
        if dur_by_chord:
            # 选择占时长最多的和弦
            chord = max(dur_by_chord.items(), key=lambda x: x[1])[0]
        else:
            chord = 'N'
        
        result.append(chord)
    
    return result


def map_to_13_chord_types(raw_label: str) -> str:
    """
    将BTC输出映射到13种和弦类型
    
    13种类型：maj, dim, sus4, m7, min, sus2, aug, dim7, maj6, hdim7, 7, m6, maj7
    
    Args:
        raw_label: BTC输出的原始和弦标签，如 'C:maj7', 'G:min', 'N'
    
    Returns:
        映射后的和弦标签，如 'C:maj7', 'G:min', 'N'
    """
    if raw_label == 'N' or raw_label == '':
        return 'N'
    
    # 解析根音和性质
    if ':' in raw_label:
        root, qual = raw_label.split(':', 1)
    else:
        # 如果没有冒号，尝试解析（如 'Cmaj7'）
        import re
        match = re.match(r'([A-Ga-g][b#]?)(.*)', raw_label)
        if match:
            root, qual = match.groups()
        else:
            return 'N'
    
    root = root.strip().upper()
    qual = qual.strip().lower()
    
    # 映射到13种和弦类型
    qual_map = {
        'maj': 'maj',
        'min': 'min',
        'dim': 'dim',
        'aug': 'aug',
        'sus4': 'sus4',
        'sus2': 'sus2',
        '7': '7',
        'dom7': '7',
        'maj7': 'maj7',
        'm7': '7',  # 注意：m7应该是minor 7th，但论文中可能是7
        'min7': 'm7',
        'dim7': 'dim7',
        'hdim7': 'hdim7',
        'm7b5': 'hdim7',
        '6': 'maj6',
        'maj6': 'maj6',
        'm6': 'm6',
        'min6': 'm6',
        # 处理一些变体
        'major': 'maj',
        'minor': 'min',
        'diminished': 'dim',
        'augmented': 'aug',
        'suspended': 'sus4',
        'sus': 'sus4',
    }
    
    # 处理特殊情况（按优先级顺序）
    if 'hdim' in qual or 'm7b5' in qual:
        mapped_qual = 'hdim7'
    elif qual.startswith('dim') and '7' in qual:
        mapped_qual = 'dim7'
    elif qual.startswith('maj') and '7' in qual:
        mapped_qual = 'maj7'
    elif qual.startswith('maj') and '6' in qual:
        mapped_qual = 'maj6'
    elif qual.startswith('min') and '7' in qual or (qual.startswith('m') and '7' in qual and 'maj' not in qual):
        mapped_qual = 'm7'
    elif qual.startswith('min') and '6' in qual or (qual.startswith('m') and '6' in qual and 'maj' not in qual):
        mapped_qual = 'm6'
    elif qual.startswith('sus') and '2' in qual:
        mapped_qual = 'sus2'
    elif qual.startswith('sus'):
        mapped_qual = 'sus4'
    else:
        mapped_qual = qual_map.get(qual, 'maj')  # 默认maj
    
    return f'{root}:{mapped_qual}'


def detect_key_from_midi(midi_path: Path) -> Optional[Tuple[str, str]]:
    """
    使用music21检测MIDI文件的调性
    
    Args:
        midi_path: MIDI文件路径
    
    Returns:
        (调性名称, 模式) 如 ('C', 'major') 或 ('A', 'minor')
    """
    if not music21_available:
        return None
    
    try:
        score = converter.parse(str(midi_path))
        
        # 使用music21的analyze方法检测调性
        keys = []
        try:
            # 方法1: 使用analyze('key')
            detected_key = score.analyze('key')
            if detected_key:
                keys.append((detected_key.tonic.name, detected_key.mode))
        except:
            pass
        
        try:
            # 方法2: 使用Krumhansl-Schmuckler算法
            from music21 import analysis
            krumhansl = analysis.discrete.KrumhanslSchmuckler()
            detected_key = krumhansl.getSolution(score)
            if detected_key:
                keys.append((detected_key.tonic.name, detected_key.mode))
        except:
            pass
        
        try:
            # 方法3: 使用Temperley-Kostka-Payne算法
            temperley = analysis.discrete.TemperleyKostkaPayne()
            detected_key = temperley.getSolution(score)
            if detected_key:
                keys.append((detected_key.tonic.name, detected_key.mode))
        except:
            pass
        
        if not keys:
            return None
        
        # 多数投票
        from collections import Counter
        key_counts = Counter(keys)
        most_common = key_counts.most_common(1)[0][0]
        return most_common
        
    except Exception as e:
        print(f"调性检测失败: {e}")
        return None


def calculate_transposition_semitones(detected_key: Tuple[str, str]) -> int:
    """
    计算需要移调的半音数（转到C大调或A小调）
    
    Args:
        detected_key: (调性名称, 模式)
    
    Returns:
        移调半音数（负数表示向下移调）
    """
    if not detected_key:
        return 0
    
    root_name, mode = detected_key
    
    # 音名到半音的映射
    note_to_semitone = {
        'C': 0, 'C#': 1, 'DB': 1, 'D': 2, 'D#': 3, 'EB': 3,
        'E': 4, 'F': 5, 'F#': 6, 'GB': 6, 'G': 7, 'G#': 8,
        'AB': 8, 'A': 9, 'A#': 10, 'BB': 10, 'B': 11
    }
    
    root_semitone = note_to_semitone.get(root_name.upper(), 0)
    
    if mode.lower() == 'major':
        # 转到C大调
        target_semitone = 0
    else:  # minor
        # 转到A小调
        target_semitone = 9  # A
    
    semitones = target_semitone - root_semitone
    
    # 确保在-12到12范围内
    if semitones > 6:
        semitones -= 12
    elif semitones < -6:
        semitones += 12
    
    return semitones


def transpose_chord(chord_label: str, semitones: int) -> str:
    """
    移调和弦的根音
    
    Args:
        chord_label: 和弦标签，如 'C:maj7'
        semitones: 移调半音数
    
    Returns:
        移调后的和弦标签
    """
    if chord_label == 'N' or ':' not in chord_label:
        return chord_label
    
    root, qual = chord_label.split(':', 1)
    
    # 音名到半音的映射
    note_to_semitone = {
        'C': 0, 'C#': 1, 'DB': 1, 'D': 2, 'D#': 3, 'EB': 3,
        'E': 4, 'F': 5, 'F#': 6, 'GB': 6, 'G': 7, 'G#': 8,
        'AB': 8, 'A': 9, 'A#': 10, 'BB': 10, 'B': 11
    }
    
    # 半音到音名的映射（优先使用#）
    semitone_to_note = {
        0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
        6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
    }
    
    root_semitone = note_to_semitone.get(root.upper(), 0)
    new_semitone = (root_semitone + semitones) % 12
    new_root = semitone_to_note[new_semitone]
    
    return f'{new_root}:{qual}'


def normalize_to_c_am(chord_sequence: List[str], midi_path: Optional[Path] = None) -> List[str]:
    """
    检测调性并归一化到C/Am
    
    Args:
        chord_sequence: 和弦序列
        midi_path: MIDI文件路径（用于调性检测）
    
    Returns:
        归一化后的和弦序列
    """
    if not music21_available or not midi_path or not midi_path.exists():
        # 如果没有music21或MIDI文件，跳过归一化
        return chord_sequence
    
    detected_key = detect_key_from_midi(midi_path)
    if not detected_key:
        return chord_sequence
    
    semitones = calculate_transposition_semitones(detected_key)
    if semitones == 0:
        return chord_sequence
    
    # 移调所有和弦
    normalized = [transpose_chord(chord, semitones) for chord in chord_sequence]
    return normalized


def process_btc_chords(wav_path: Path, output_dir: Path, video_id: str,
                       btc_python: str, btc_repo_path: Path,
                       midi_path: Optional[Path] = None,
                       normalize_key: bool = True) -> Tuple[bool, Optional[Path]]:
    """
    完整的BTC和弦处理流程
    
    Args:
        wav_path: WAV文件路径
        output_dir: 输出目录
        video_id: 视频ID
        btc_python: BTC环境的Python路径
        btc_repo_path: BTC仓库路径
        midi_path: MIDI文件路径（用于调性归一化，可选）
        normalize_key: 是否进行调性归一化
    
    Returns:
        (成功标志, 输出lab文件路径)
    """
    # 1. 运行BTC推理
    success, btc_lab_path = run_btc_inference(wav_path, output_dir, video_id, 
                                             btc_python, btc_repo_path)
    if not success or not btc_lab_path:
        return False, None
    
    # 2. 解析lab文件
    lab_events = parse_lab_file(btc_lab_path)
    if not lab_events:
        return False, None
    
    # 3. 计算总时长
    total_duration = max(et for _, et, _ in lab_events) if lab_events else 0.0
    
    # 4. 重采样到1Hz
    chord_sequence = resample_to_1hz(lab_events, total_duration)
    
    # 5. 映射到13种和弦类型
    mapped_sequence = [map_to_13_chord_types(chord) for chord in chord_sequence]
    
    # 6. 调性归一化（可选）
    if normalize_key and midi_path and midi_path.exists():
        mapped_sequence = normalize_to_c_am(mapped_sequence, midi_path)
    
    # 7. 写入输出文件
    output_dir.mkdir(parents=True, exist_ok=True)
    output_lab_path = output_dir / f'{video_id}.lab'
    
    with open(output_lab_path, 'w', encoding='utf-8') as f:
        f.write('key C major\n')  # 默认C大调（归一化后）
        for i, chord in enumerate(mapped_sequence):
            f.write(f'{i} {chord}\n')
    
    return True, output_lab_path

