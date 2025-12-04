# 模型生成使用说明

## 概述

`generate.py` 是用于生成视频-音乐同步结果的脚本。它会：
1. 加载训练好的模型
2. 从视频中提取特征
3. 生成对应的和弦序列
4. 将生成的和弦转换为 MIDI 和音频
5. 将生成的音频合成到原始视频中

## 使用方法

### 方法 1: 使用启动脚本（推荐）

```bash
./run_generate.sh
```

启动脚本会自动设置必要的环境变量（LD_PRELOAD）来解决 libffi 兼容性问题。

### 方法 2: 手动运行

```bash
cd /home/jim/Video2Music
conda activate v2m_train
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
python generate.py
```

## 配置参数

在 `generate.py` 中可以修改以下参数：

- `test_id`: 测试视频ID（默认：`"049"`）
- `num_prime_chord`: Primer和弦数量（默认：30）
- `is_voice`: 是否使用人声（默认：True）
- `isArp`: 是否使用琶音（默认：True）
- `duration`: 每个和弦的持续时间（秒，默认：2）
- `tempo`: 节拍速度（默认：120）
- `octave`: 八度（默认：4）
- `velocity`: 音符力度（默认：100）

## 输出文件

生成的文件会保存在以下位置：

- MIDI文件: `output/<test_id>/<test_id>_gen.mid`
- 音频文件（FLAC）: `output/<test_id>/<test_id>_gen.flac`
- 最终视频: `output/<test_id>/<test_id>_final.mp4`

## 依赖要求

- Python 环境: `v2m_train`
- 已安装的包:
  - PyTorch
  - moviepy
  - pretty_midi
  - midi2audio
  - FluidSynth

## 常见问题

### 1. ModuleNotFoundError: No module named 'moviepy.editor'

**原因**: moviepy 2.x 版本不再有 `editor` 子模块

**解决**: 已修复，现在直接从 `moviepy` 导入所需类

### 2. OSError: undefined symbol: ffi_type_uint32

**原因**: libffi 版本兼容性问题

**解决**: 使用 `run_generate.sh` 脚本，或手动设置 `LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7`

### 3. 模型文件未找到

**原因**: 模型权重文件路径不正确

**解决**: 检查 `saved_models/` 目录下是否有训练好的模型文件

## 注意事项

1. **环境变量**: 必须设置 `LD_PRELOAD` 才能正常运行（pretty_midi 需要）
2. **GPU**: 如果有 GPU，模型会自动使用 GPU 加速
3. **视频文件**: 确保测试视频文件存在于 `dataset/vevo/<test_id>.mp4`
4. **模型权重**: 确保已训练好模型并保存在正确的位置

## 示例

```bash
# 使用默认参数生成视频 049 的音乐
./run_generate.sh

# 修改 test_id 后生成其他视频
# 编辑 generate.py，修改 test_id = "001"
./run_generate.sh
```


