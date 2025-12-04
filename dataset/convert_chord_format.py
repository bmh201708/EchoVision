#!/usr/bin/env python3
"""
将你的数据集和弦格式转换为官方数据集格式

转换规则：
- "C:maj" -> "C" (无属性表示大三和弦)
- "C:min" -> "C:min" (保持不变)
- "N" -> "N" (保持不变)
"""
import json
from pathlib import Path

# 官方数据集的完整 chord.json
OFFICIAL_CHORD_DICT = {
    "N": 0, "C": 1, "C:dim": 2, "C:sus4": 3, "C:min7": 4, "C:min": 5, 
    "C:sus2": 6, "C:aug": 7, "C:dim7": 8, "C:maj6": 9, "C:hdim7": 10, 
    "C:7": 11, "C:min6": 12, "C:maj7": 13, "C#": 14, "C#:dim": 15, 
    "C#:sus4": 16, "C#:min7": 17, "C#:min": 18, "C#:sus2": 19, "C#:aug": 20, 
    "C#:dim7": 21, "C#:maj6": 22, "C#:hdim7": 23, "C#:7": 24, "C#:min6": 25, 
    "C#:maj7": 26, "D": 27, "D:dim": 28, "D:sus4": 29, "D:min7": 30, 
    "D:min": 31, "D:sus2": 32, "D:aug": 33, "D:dim7": 34, "D:maj6": 35, 
    "D:hdim7": 36, "D:7": 37, "D:min6": 38, "D:maj7": 39, "D#": 40, 
    "D#:dim": 41, "D#:sus4": 42, "D#:min7": 43, "D#:min": 44, "D#:sus2": 45, 
    "D#:aug": 46, "D#:dim7": 47, "D#:maj6": 48, "D#:hdim7": 49, "D#:7": 50, 
    "D#:min6": 51, "D#:maj7": 52, "E": 53, "E:dim": 54, "E:sus4": 55, 
    "E:min7": 56, "E:min": 57, "E:sus2": 58, "E:aug": 59, "E:dim7": 60, 
    "E:maj6": 61, "E:hdim7": 62, "E:7": 63, "E:min6": 64, "E:maj7": 65, 
    "F": 66, "F:dim": 67, "F:sus4": 68, "F:min7": 69, "F:min": 70, 
    "F:sus2": 71, "F:aug": 72, "F:dim7": 73, "F:maj6": 74, "F:hdim7": 75, 
    "F:7": 76, "F:min6": 77, "F:maj7": 78, "F#": 79, "F#:dim": 80, 
    "F#:sus4": 81, "F#:min7": 82, "F#:min": 83, "F#:sus2": 84, "F#:aug": 85, 
    "F#:dim7": 86, "F#:maj6": 87, "F#:hdim7": 88, "F#:7": 89, "F#:min6": 90, 
    "F#:maj7": 91, "G": 92, "G:dim": 93, "G:sus4": 94, "G:min7": 95, 
    "G:min": 96, "G:sus2": 97, "G:aug": 98, "G:dim7": 99, "G:maj6": 100, 
    "G:hdim7": 101, "G:7": 102, "G:min6": 103, "G:maj7": 104, "G#": 105, 
    "G#:dim": 106, "G#:sus4": 107, "G#:min7": 108, "G#:min": 109, "G#:sus2": 110, 
    "G#:aug": 111, "G#:dim7": 112, "G#:maj6": 113, "G#:hdim7": 114, "G#:7": 115, 
    "G#:min6": 116, "G#:maj7": 117, "A": 118, "A:dim": 119, "A:sus4": 120, 
    "A:min7": 121, "A:min": 122, "A:sus2": 123, "A:aug": 124, "A:dim7": 125, 
    "A:maj6": 126, "A:hdim7": 127, "A:7": 128, "A:min6": 129, "A:maj7": 130, 
    "A#": 131, "A#:dim": 132, "A#:sus4": 133, "A#:min7": 134, "A#:min": 135, 
    "A#:sus2": 136, "A#:aug": 137, "A#:dim7": 138, "A#:maj6": 139, "A#:hdim7": 140, 
    "A#:7": 141, "A#:min6": 142, "A#:maj7": 143, "B": 144, "B:dim": 145, 
    "B:sus4": 146, "B:min7": 147, "B:min": 148, "B:sus2": 149, "B:aug": 150, 
    "B:dim7": 151, "B:maj6": 152, "B:hdim7": 153, "B:7": 154, "B:min6": 155, 
    "B:maj7": 156
}

def convert_chord_format(your_chord):
    """将你的格式转换为官方格式"""
    if your_chord == "N":
        return "N"
    
    # 解析 "C:maj" -> ("C", "maj")
    if ":" in your_chord:
        root, attr = your_chord.split(":", 1)
        if attr == "maj":
            # "C:maj" -> "C" (官方格式中，无属性就是大三和弦)
            return root
        else:
            # "C:min" -> "C:min" (其他属性保持不变)
            return your_chord
    else:
        # 如果已经是官方格式（如 "C"），保持不变
        return your_chord

def main():
    dataset_root = Path(__file__).parent
    meta_dir = dataset_root / 'vevo_meta'
    chord_file = meta_dir / 'chord.json'
    
    # 读取你的 chord.json
    if not chord_file.exists():
        print(f"错误: 文件不存在: {chord_file}")
        return
    
    with open(chord_file, 'r', encoding='utf-8') as f:
        your_chord_dict = json.load(f)
    
    print(f"你的数据集有 {len(your_chord_dict)} 种和弦")
    
    # 创建转换后的字典（使用官方格式和ID）
    converted_dict = {}
    id_mapping = {}  # 你的ID -> 官方ID
    
    for your_chord, your_id in your_chord_dict.items():
        official_chord = convert_chord_format(your_chord)
        
        if official_chord in OFFICIAL_CHORD_DICT:
            official_id = OFFICIAL_CHORD_DICT[official_chord]
            converted_dict[official_chord] = official_id
            id_mapping[your_id] = official_id
            print(f"  {your_chord} (ID {your_id}) -> {official_chord} (ID {official_id})")
        else:
            print(f"  警告: {official_chord} 不在官方词汇表中")
    
    # 备份原文件
    backup_file = chord_file.with_suffix('.json.backup')
    print(f"\n备份原文件到: {backup_file}")
    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(your_chord_dict, f, indent=2, ensure_ascii=False)
    
    # 保存转换后的 chord.json（使用官方完整字典）
    print(f"\n更新 chord.json 为官方格式（包含所有157种和弦）...")
    with open(chord_file, 'w', encoding='utf-8') as f:
        json.dump(OFFICIAL_CHORD_DICT, f, indent=2, ensure_ascii=False)
    
    # 保存ID映射（用于转换.lab文件中的ID）
    mapping_file = meta_dir / 'chord_id_mapping.json'
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(id_mapping, f, indent=2, ensure_ascii=False)
    
    print(f"\n完成！")
    print(f"  - 转换了 {len(id_mapping)} 个和弦")
    print(f"  - chord.json 已更新为官方格式（157种和弦）")
    print(f"  - ID映射已保存到: {mapping_file}")
    print(f"\n注意: 你还需要转换 .lab 文件中的和弦ID！")

if __name__ == '__main__':
    main()


