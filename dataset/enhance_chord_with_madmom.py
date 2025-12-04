#!/usr/bin/env python3
"""
使用 madmom 增强和弦识别（支持更多和弦类型）

安装：
    pip install madmom

注意：madmom 可能识别出更多和弦类型，但需要确保输出格式与官方数据集兼容
"""
import os
import sys
from pathlib import Path

try:
    from madmom.features.chords import (
        CNNChordFeatureProcessor,
        CRFChordRecognitionProcessor
    )
except ImportError:
    print("错误: 请先安装 madmom")
    print("  pip install madmom")
    sys.exit(1)

def recognize_chords_madmom(wav_path, output_dir, video_id):
    """
    使用 madmom 识别和弦
    
    Args:
        wav_path: WAV 文件路径
        output_dir: 输出目录
        video_id: 视频 ID
    
    Returns:
        bool: 是否成功
    """
    try:
        # 初始化处理器
        feature_processor = CNNChordFeatureProcessor()
        chord_processor = CRFChordRecognitionProcessor()
        
        # 提取特征
        print(f"  [{video_id}] 提取音频特征...")
        features = feature_processor(str(wav_path))
        
        # 识别和弦
        print(f"  [{video_id}] 识别和弦...")
        chords = chord_processor(features)
        
        # 转换格式：madmom 输出 -> CSV 格式
        csv_path = Path(output_dir) / f'{video_id}.csv'
        lab_path = Path(output_dir) / f'{video_id}.lab'
        
        # 写入 CSV
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('chord,start,end\n')
            for start, end, chord in chords:
                f.write(f'{chord},{start:.6f},{end:.6f}\n')
        
        # 转换为 .lab 格式（每秒一行）
        import math
        lines = ['key C major']
        
        if chords:
            total_dur = int(math.ceil(chords[-1][1]))
            for sec in range(total_dur):
                label = 'N'
                for start, end, chord in chords:
                    if start <= sec < end:
                        label = chord or 'N'
                        break
                lines.append(f'{sec} {label}')
        else:
            lines.append('0 N')
        
        with open(lab_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        
        print(f"  [{video_id}] ✓ madmom 和弦识别完成")
        print(f"  [{video_id}]   识别到 {len(chords)} 个和弦片段")
        
        # 统计和弦类型
        chord_types = set()
        for _, _, chord in chords:
            if chord and chord != 'N':
                chord_types.add(chord)
        
        print(f"  [{video_id}]   和弦类型: {sorted(chord_types)}")
        
        return True
        
    except Exception as e:
        print(f"  [{video_id}] ✗ madmom 和弦识别失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='使用 madmom 识别和弦')
    parser.add_argument('wav_path', help='WAV 文件路径')
    parser.add_argument('output_dir', help='输出目录')
    parser.add_argument('video_id', help='视频 ID')
    
    args = parser.parse_args()
    
    recognize_chords_madmom(
        Path(args.wav_path),
        Path(args.output_dir),
        args.video_id
    )



