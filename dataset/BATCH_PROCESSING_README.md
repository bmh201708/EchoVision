# 批量数据集处理脚本使用说明

## 概述

`batch_prepare_dataset.py` 是一个批量处理脚本，可以自动处理 `dataset/vevo/` 目录下的所有 MP4 文件，提取所需的各种特征。

## 功能

脚本会自动执行以下处理步骤（按顺序）：

1. **抽帧** - 每秒抽取 1 帧，保存到 `vevo_frame/<id>/`
2. **运动特征** - 计算运动特征，保存到 `vevo_motion/all/<id>.lab`
3. **语义特征 (CLIP)** - 提取 CLIP 特征，保存到 `vevo_semantic/all/2d/clip_l14p/<id>.npy`
4. **情感特征** - 提取 6 类情感特征，保存到 `vevo_emotion/6c_l14p/all/<id>.lab`
5. **分镜 + Scene Offset** - 检测场景切换，保存到 `vevo_scene/all/<id>.lab` 和 `vevo_scene_offset/all/<id>.lab`
6. **音频提取** - 从 MP4 提取 WAV，保存到 `vevo_audio/wav/<id>.wav`
7. **响度特征** - 计算响度特征，保存到 `vevo_loudness/all/<id>.lab`
8. **和弦识别 (Omnizart)** - 使用 Omnizart 识别和弦，保存到 `vevo_chord/lab_v2_norm/all/<id>.lab`
9. **MIDI 生成** - 根据和弦标注生成 MIDI，保存到 `vevo_midi/all/<id>.mid`
10. **Note Density** - 计算音符密度，保存到 `vevo_note_density/all/<id>.lab`
11. **更新元数据** - 在所有视频处理完成后，更新 `vevo_meta/` 下的元数据文件

## 使用方法

### 基本用法

处理所有视频文件：
```bash
cd /home/jim/Video2Music/dataset
python batch_prepare_dataset.py
```

### 常用选项

**跳过已存在的文件**（用于断点续传）：
```bash
python batch_prepare_dataset.py --skip-existing
```

**只处理指定的视频**：
```bash
python batch_prepare_dataset.py --video-id 049
```

**跳过某些处理步骤**（例如跳过和弦识别和 MIDI 生成）：
```bash
python batch_prepare_dataset.py --skip-steps chord,midi
```

**组合使用**：
```bash
# 跳过已存在的文件，只处理视频 049，跳过和弦识别
python batch_prepare_dataset.py --skip-existing --video-id 049 --skip-steps chord
```

## 可跳过的步骤

使用 `--skip-steps` 参数时，可以跳过以下步骤（用逗号分隔）：

- `frames` - 抽帧
- `motion` - 运动特征
- `semantic` - 语义特征
- `emotion` - 情感特征
- `scene` - 分镜
- `audio` - 音频提取
- `loudness` - 响度特征
- `chord` - 和弦识别
- `midi` - MIDI 生成
- `note_density` - Note Density
- `metadata` - 元数据更新

## 日志

脚本会生成日志文件 `batch_prepare_dataset.log`，记录所有处理过程和错误信息。

## 注意事项

1. **环境要求**：
   - 需要安装所有依赖（`pip install -r ../requirements.txt`）
   - 需要安装 `omnizart` 并下载 checkpoint（在 `omni` 环境中）
   - 需要 `ffmpeg` 命令可用

2. **Omnizart 环境**：
   - 脚本会使用 `/home/jim/anaconda3/envs/omni/bin/python` 运行 Omnizart
   - 如果路径不同，需要修改脚本中的 `OMNI_PYTHON` 变量

3. **GPU 支持**：
   - CLIP 模型会自动使用 GPU（如果可用）
   - 如果没有 GPU，会自动回退到 CPU（速度较慢）

4. **错误处理**：
   - 如果某个视频的某个步骤失败，脚本会继续处理下一个视频
   - 失败的视频会在最后统计中显示

5. **元数据更新**：
   - 元数据更新会在所有视频处理完成后自动执行
   - 会重新扫描所有已处理的和弦标注文件，生成字典和 split 文件

## 示例输出

```
2025-12-04 01:30:00 - INFO - 找到 100 个视频文件
处理视频: 001
  [001] ✓ 抽帧完成
  [001] ✓ 运动特征完成
  [001] ✓ 语义特征完成
  ...
  [001] ✓ 处理完成
处理视频: 002
  ...
============================================================
处理完成!
  成功: 98
  失败: 2
  失败的视频: 045, 067
============================================================
```

## 与 Notebook 的关系

- `prepare_dataset.ipynb` - 用于交互式处理单个视频，适合调试和测试
- `batch_prepare_dataset.py` - 用于批量处理多个视频，适合生产环境

两个工具使用相同的处理逻辑，确保结果一致。




