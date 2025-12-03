#!/usr/bin/env python3
"""提取 videomaterial.zip 中的 MP4 文件到 dataset/vevo/"""
import zipfile
import os
import shutil
from pathlib import Path

zip_path = Path('/home/jim/videomaterial.zip')
target_dir = Path('/home/jim/Video2Music/dataset/vevo')

# 检查文件是否存在
if not zip_path.exists():
    print(f"错误: 文件不存在: {zip_path}")
    exit(1)

print(f"正在处理: {zip_path}")

# 确保目标目录存在
target_dir.mkdir(parents=True, exist_ok=True)

# 打开 zip 文件
with zipfile.ZipFile(zip_path, 'r') as z:
    # 找到所有 MP4 文件（排除 __MACOSX 系统文件）
    mp4_files = [f for f in z.namelist() if f.endswith('.mp4') and not f.startswith('__MACOSX')]
    
    print(f"找到 {len(mp4_files)} 个 MP4 文件\n")
    
    # 提取并移动 MP4 文件
    moved_count = 0
    skipped_count = 0
    for mp4_file in mp4_files:
        # 获取文件名（不含路径）
        filename = os.path.basename(mp4_file)
        target_path = target_dir / filename
        
        # 如果文件已存在，跳过
        if target_path.exists():
            print(f"  跳过（已存在）: {filename}")
            skipped_count += 1
            continue
        
        # 提取文件
        try:
            with z.open(mp4_file) as source:
                with open(target_path, 'wb') as target:
                    shutil.copyfileobj(source, target)
            file_size = target_path.stat().st_size / (1024 * 1024)  # MB
            print(f"  ✓ 已移动: {filename} ({file_size:.1f} MB)")
            moved_count += 1
        except Exception as e:
            print(f"  ✗ 错误: {filename} - {e}")

print(f"\n完成！")
print(f"  成功移动: {moved_count} 个文件")
print(f"  跳过（已存在）: {skipped_count} 个文件")
print(f"  目标目录: {target_dir}")

# 列出所有视频文件
all_videos = sorted(target_dir.glob('*.mp4'))
print(f"\n当前共有 {len(all_videos)} 个视频文件:")
for v in all_videos:
    size_mb = v.stat().st_size / (1024 * 1024)
    print(f"  {v.name} ({size_mb:.1f} MB)")

