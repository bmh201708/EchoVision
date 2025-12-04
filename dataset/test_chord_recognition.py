#!/usr/bin/env python3
"""
测试不同和弦识别工具的输出格式和和弦类型
"""
import sys
from pathlib import Path

def test_madmom(wav_path):
    """测试 madmom"""
    try:
        from madmom.features.chords import (
            CNNChordFeatureProcessor,
            CRFChordRecognitionProcessor
        )
        
        print("=" * 60)
        print("测试 madmom 和弦识别")
        print("=" * 60)
        
        feature_processor = CNNChordFeatureProcessor()
        chord_processor = CRFChordRecognitionProcessor()
        
        print("提取特征...")
        features = feature_processor(str(wav_path))
        
        print("识别和弦...")
        chords = chord_processor(features)
        
        print(f"\n识别到 {len(chords)} 个和弦片段")
        
        # 统计和弦类型
        chord_types = {}
        for start, end, chord in chords[:50]:  # 只看前50个
            if chord and chord != 'N':
                chord_types[chord] = chord_types.get(chord, 0) + 1
        
        print(f"\n和弦类型统计（前50个片段）:")
        for chord, count in sorted(chord_types.items(), key=lambda x: -x[1]):
            print(f"  {chord}: {count} 次")
        
        # 分析属性
        attrs = set()
        for chord in chord_types.keys():
            if ':' in chord:
                attr = chord.split(':')[1]
                attrs.add(attr)
            elif chord != 'N':
                attrs.add('(无属性，即maj)')
        
        print(f"\n识别的属性类型: {sorted(attrs)}")
        
        # 显示前10个和弦
        print(f"\n前10个和弦片段:")
        for i, (start, end, chord) in enumerate(chords[:10]):
            print(f"  {i+1}. [{start:.2f}-{end:.2f}s] {chord}")
        
        return True
        
    except ImportError:
        print("madmom 未安装")
        return False
    except Exception as e:
        print(f"madmom 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_librosa_chord(wav_path):
    """测试 librosa 和弦识别（简单版本）"""
    try:
        import librosa
        import numpy as np
        
        print("\n" + "=" * 60)
        print("测试 librosa 和弦识别（基于色度特征）")
        print("=" * 60)
        
        # 加载音频
        y, sr = librosa.load(str(wav_path), duration=30)  # 只处理前30秒
        
        # 提取色度特征
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # 简单的和弦识别（基于色度向量）
        # 这里只是示例，实际需要更复杂的算法
        print(f"提取了色度特征: {chroma.shape}")
        print("注意: librosa 本身不提供和弦识别，需要额外的算法")
        
        return False
        
    except ImportError:
        print("librosa 未安装")
        return False
    except Exception as e:
        print(f"librosa 测试失败: {e}")
        return False

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python test_chord_recognition.py <wav_file>")
        sys.exit(1)
    
    wav_path = Path(sys.argv[1])
    if not wav_path.exists():
        print(f"错误: 文件不存在: {wav_path}")
        sys.exit(1)
    
    print(f"测试文件: {wav_path}")
    print(f"文件大小: {wav_path.stat().st_size / 1024 / 1024:.1f} MB\n")
    
    # 测试 madmom
    madmom_ok = test_madmom(wav_path)
    
    # 测试 librosa（如果需要）
    # test_librosa_chord(wav_path)
    
    if not madmom_ok:
        print("\n建议:")
        print("1. 尝试安装 madmom: pip install madmom")
        print("2. 或使用 conda: conda install -c conda-forge madmom")
        print("3. 或继续使用 Omnizart（只识别 maj/min）")

