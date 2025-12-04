# 视频-音乐同步数据集构建报告

## 1. 项目概述

本项目旨在构建一个用于视频-音乐同步任务的数据集，通过从视频中提取多模态特征（视觉、音频、音乐），训练模型学习视频内容与音乐和弦之间的对应关系。

### 1.1 研究目标

- 构建一个包含视频和对应音乐标注的数据集
- 提取视频的视觉特征（语义、情感、运动、场景）
- 提取音频的音乐特征（和弦、响度、音符密度）
- 建立视频特征与音乐特征的映射关系

### 1.2 数据集规模

- **视频总数**: 100 个 MP4 文件（`dataset/vevo/`目录）
- **已处理视频**: 100 个（已完成所有特征提取）
- **数据集划分**:
  - 训练集: 80 个视频（80%）
  - 验证集: 10 个视频（10%）
  - 测试集: 10 个视频（10%）
  - 划分比例：8:1:1（随机划分）
- **和弦类型**: 
  - **实际识别**: 109 种不同和弦类型
  - **和弦属性**: 14 种（maj, min, dim, aug, sus4, sus2, 7, maj7, m7, dim7, maj6, m6, hdim7, N）
  - **根音**: 12 个（C, C#, D, D#, E, F, F#, G, G#, A, A#, B）

## 2. 数据集构建流程

### 2.1 数据准备阶段

#### 2.1.1 视频文件收集
- 收集了 100 个音乐视频文件（MP4 格式）
- 文件命名规范：`<id>.mp4`（如 `001.mp4`, `002.mp4`）
- 视频存储位置：`dataset/vevo/`
- **当前状态**: ✅ 已完成所有 100 个视频的特征提取

#### 2.1.2 环境配置
- **Python 环境**: 
  - `v2m_train`: 主训练环境，包含 PyTorch、CLIP 等依赖
  - `omni`: Omnizart 和弦识别环境
- **关键依赖**:
  - PyTorch, CLIP, OpenCV, scenedetect
  - Omnizart（和弦识别）
  - ffmpeg（视频/音频处理）

### 2.2 特征提取流程

数据集构建采用自动化批量处理脚本 `batch_prepare_dataset.py`，**严格按照以下顺序**执行**11个特征提取步骤**：

> **⚠️ 重要提示**: 
> - 处理步骤有**严格的依赖关系**，必须按顺序执行
> - 前面的步骤失败会导致后续步骤无法进行
> - 每个视频都会完整执行步骤1→11
> - 所有视频处理完成后，自动执行步骤11更新元数据

#### 2.2.1 视觉特征提取

**步骤 1: 视频抽帧** ⬇️
- **方法**: 使用 ffmpeg，每秒抽取 1 帧
- **输出**: `vevo_frame/<id>/<id>_001.jpg`, `<id>_002.jpg`, ...
- **依赖**: 需要原始 MP4 视频文件
- **用途**: 后续语义和情感特征提取的基础
- **耗时**: ~1-2秒/视频

**步骤 2: 运动特征（Motion）** ⬇️
- **方法**: 计算相邻帧之间的光流或运动向量
- **输出**: `vevo_motion/all/<id>.lab`
- **格式**: 每行 `时间(秒) 运动值`
- **依赖**: 需要步骤1的抽帧结果
- **用途**: 捕捉视频的动态变化
- **耗时**: ~1-2秒/视频

**步骤 3: 语义特征（Semantic）** ⬇️
- **方法**: 使用 CLIP (ViT-L/14@336px) 提取图像特征
- **输出**: `vevo_semantic/all/2d/clip_l14p/<id>.npy`
- **维度**: 768 维特征向量
- **依赖**: 需要步骤1的抽帧结果
- **用途**: 理解视频的语义内容
- **耗时**: ~5-10秒/视频（取决于视频长度）

**步骤 4: 情感特征（Emotion）** ⬇️
- **方法**: 使用 CLIP 模型，结合 6 类情感标签（exciting, fearful, tense, sad, relaxing, neutral）
- **输出**: `vevo_emotion/6c_l14p/all/<id>.lab`
- **格式**: 每行 `时间 6个情感概率值`
- **依赖**: 需要步骤1的抽帧结果
- **用途**: 识别视频的情感色彩
- **耗时**: ~5-10秒/视频（取决于视频长度）

**步骤 5: 场景检测（Scene Detection）** ⬇️
- **方法**: 使用 scenedetect 库的 AdaptiveDetector
- **输出**:
  - `vevo_scene/all/<id>.lab`: 场景ID序列
  - `vevo_scene_offset/all/<id>.lab`: 场景偏移量
- **依赖**: 需要原始 MP4 视频文件
- **用途**: 识别视频的场景切换点
- **耗时**: ~1-2秒/视频

#### 2.2.2 音频特征提取

**步骤 6: 音频提取** ⬇️
- **方法**: 使用 ffmpeg 从 MP4 提取 WAV 音频
- **参数**: 44.1kHz, 单声道, PCM 16-bit
- **输出**: `vevo_audio/wav/<id>.wav`
- **依赖**: 需要原始 MP4 视频文件
- **用途**: 后续所有音频特征提取的基础
- **耗时**: ~1-2秒/视频

**步骤 7: 响度特征（Loudness）** ⬇️
- **方法**: 计算音频的 RMS 响度
- **输出**: `vevo_loudness/all/<id>.lab`
- **格式**: 每行 `时间(秒) 响度值`
- **依赖**: 需要步骤6的WAV音频文件
- **用途**: 捕捉音乐的动态变化
- **耗时**: ~1秒/视频

**步骤 8: 和弦识别（Chord Recognition）** ⬇️

数据集构建支持两种和弦识别方法：

**方法1: Omnizart（默认）**
- **工具**: Omnizart（基于 Harmony Transformer）
- **方法**: 使用预训练的 chord_v1 模型
- **输出**: 
  - `vevo_chord/lab_v2_norm/all/<id>.lab`: 和弦标注文件
  - CSV 格式：`Chord,Start,End`
- **和弦类型**: 25 种
  - 12 个大三和弦（C:maj, C#:maj, ..., B:maj）
  - 12 个小三和弦（C:min, C#:min, ..., B:min）
  - 1 个无和弦（N）
- **限制**: Omnizart 只支持识别 maj/min 两种属性，不支持更复杂的和弦（如 7、maj7、dim、aug 等）

**方法2: BTC-ISMIR19（推荐）**
- **工具**: BTC（Bi-directional Transformer for Chord Recognition, Park et al. 2019）
- **方法**: 使用预训练的 BTC 模型，支持大词表（`--voca True`）
- **处理流程**:
  1. BTC模型推理：生成带时间戳的和弦标签（`.lab`格式）
  2. 重采样到1Hz：每秒一个和弦，选择该秒内占时长最多的和弦
  3. 映射到13种和弦类型：将BTC输出映射到标准13种和弦属性
  4. 调性归一化（可选）：检测调性并归一化到C大调或A小调
- **输出**: 
  - `vevo_chord/lab_v2_norm/all/<id>.lab`: 归一化后的和弦标注文件
- **和弦类型**: 13种属性 + 12个根音 = 多种组合
  - **13种和弦属性**: maj, min, dim, aug, sus4, sus2, 7, maj7, m7, dim7, maj6, m6, hdim7
  - **12个根音**: C, C#, D, D#, E, F, F#, G, G#, A, A#, B
  - **无和弦**: N
- **优势**: 
  - 支持更丰富的和弦类型（7和弦、maj7、dim、sus等）
  - 更符合论文要求的13种和弦类型
  - 识别精度高，适合复杂音乐

**步骤 9: MIDI 生成** ⬇️
- **方法**: 根据和弦标注生成 MIDI 文件
- **输出**: `vevo_midi/all/<id>.mid`
- **依赖**: 需要步骤8的和弦标注文件
- **用途**: 可听化的音乐输出
- **耗时**: ~1秒/视频

**步骤 10: 音符密度（Note Density）** ⬇️
- **方法**: 从 MIDI 文件计算音符密度
- **输出**: `vevo_note_density/all/<id>.lab`
- **格式**: 每行 `时间(秒) 音符数量`
- **依赖**: 需要步骤9的MIDI文件
- **用途**: 捕捉音乐的节奏密度
- **耗时**: ~1秒/视频

**步骤 11: 元数据更新** ⬇️
- **方法**: 扫描所有已处理的视频，生成/更新元数据文件
- **输出**: 
  - `vevo_meta/chord.json`: 和弦字典
  - `vevo_meta/chord_attr.json`: 和弦属性字典
  - `vevo_meta/chord_root.json`: 根音字典
  - `vevo_meta/split/v1/train.txt`, `val.txt`, `test.txt`: 数据集划分
  - `vevo_meta/idlist.txt`: 视频ID列表
- **依赖**: 需要所有前面的步骤完成
- **用途**: 统一管理数据集元数据，供训练和评估使用
- **耗时**: ~1-2秒（一次性处理）

### 2.3 处理流程图

```
原始MP4视频文件
    ↓
[步骤1] 视频抽帧 → vevo_frame/
    ↓
[步骤2] 运动特征 → vevo_motion/
    ↓
[步骤3] 语义特征 → vevo_semantic/
    ↓
[步骤4] 情感特征 → vevo_emotion/
    ↓
[步骤5] 场景检测 → vevo_scene/
    ↓
[步骤6] 音频提取 → vevo_audio/
    ↓
[步骤7] 响度特征 → vevo_loudness/
    ↓
[步骤8] 和弦识别 → vevo_chord/
    ↓
[步骤9] MIDI生成 → vevo_midi/
    ↓
[步骤10] 音符密度 → vevo_note_density/
    ↓
[步骤11] 元数据更新 → vevo_meta/
    ↓
完成！数据集可用于训练
```

### 2.4 元数据生成

**步骤11**会自动生成以下元数据文件：

**1. 和弦字典**
- `chord.json`: 和弦名称到ID的映射（动态生成，根据实际识别结果）
- `chord_inv.json`: ID到和弦名称的逆映射
- `chord_root.json`: 根音字典（12个根音 + N）
- `chord_attr.json`: 属性字典（14种属性）
  - 基础13种：maj, min, dim, aug, sus4, sus2, 7, maj7, m7, dim7, maj6, m6, hdim7
  - 无和弦：N

**2. 数据集划分**
- `split/v1/train.txt`: 训练集视频ID列表
- `split/v1/val.txt`: 验证集视频ID列表
- `split/v1/test.txt`: 测试集视频ID列表
- **划分方式**:
  - **默认模式**: 8:1:1（随机划分所有视频）
  - **增量模式** (`--preserve-splits`): 保持现有划分，新视频按8:1:1分配

**3. 其他元数据**
- `idlist.txt`: 所有视频ID列表
- `top_chord.txt`: 最常用的和弦统计

## 3. 技术实现细节

### 3.1 批量处理脚本

**脚本名称**: `batch_prepare_dataset.py`

**核心功能**: 自动化执行上述**11个处理步骤**，按顺序处理所有视频文件

**处理流程**:
1. **扫描视频**: 自动扫描 `dataset/vevo/` 目录下的所有 MP4 文件
2. **逐个处理**: 对每个视频，严格按照步骤1→11的顺序执行
3. **步骤依赖**: 每个步骤依赖前一步的输出，必须按顺序执行
4. **元数据更新**: 所有视频处理完成后，自动执行步骤11更新元数据

**主要功能**:
- ✅ **自动扫描**: 扫描 `dataset/vevo/` 目录下的所有 MP4 文件
- ✅ **顺序执行**: **严格按照步骤1→11的顺序**执行所有特征提取步骤
- ✅ **断点续传**: 支持 `--skip-existing`，跳过已存在的输出文件
- ✅ **选择性处理**: 支持 `--video-id` 处理单个视频，`--skip-steps` 跳过指定步骤
- ✅ **增量处理**: 支持 `--preserve-splits`，保持现有数据集划分，新视频按8:1:1分配
- ✅ **多方法和弦识别**: 支持 `--chord-method`，可选择 `omnizart` 或 `btc`
- ✅ **错误处理**: 自动记录错误日志，失败的视频不会影响其他视频的处理
- ✅ **进度显示**: 使用 tqdm 显示处理进度

**关键特性**:
- **跨环境调用**: 
  - 在 `v2m_train` 环境中调用 `omni` 环境的 Python 来运行 Omnizart
  - 在 `v2m_train` 环境中调用 `btc` 环境的 Python 来运行 BTC
- **环境变量处理**: 自动设置 `LD_PRELOAD` 解决 libffi 兼容性问题
- **进度跟踪**: 使用 tqdm 显示处理进度
- **增量更新**: 自动更新 `chord.json` 和 `chord_attr.json`，保持现有ID不变，新和弦追加

**命令行参数**:
```bash
python batch_prepare_dataset.py [选项]

选项:
  --skip-existing: 跳过已存在的输出文件（推荐用于断点续传）
  --video-id ID: 只处理指定的视频 ID（如：001）
  --skip-steps STEPS: 跳过指定的处理步骤（用逗号分隔，如：frames,motion）
                      可用步骤名：frames, motion, semantic, emotion, scene, 
                                 audio, loudness, chord, midi, note_density, metadata
  --preserve-splits: 保持现有的训练/验证/测试集划分，新视频按8:1:1比例分配
  --chord-method METHOD: 和弦识别方法（omnizart 或 btc，默认：omnizart）
  --max-workers N: 并行处理的视频数量（默认：1，串行处理）
```

**处理步骤名称对照表**:
| 步骤编号 | 步骤名称 | 说明 |
|---------|---------|------|
| 1 | `frames` | 视频抽帧 |
| 2 | `motion` | 运动特征 |
| 3 | `semantic` | 语义特征 |
| 4 | `emotion` | 情感特征 |
| 5 | `scene` | 场景检测 |
| 6 | `audio` | 音频提取 |
| 7 | `loudness` | 响度特征 |
| 8 | `chord` | 和弦识别 |
| 9 | `midi` | MIDI生成 |
| 10 | `note_density` | 音符密度 |
| 11 | `metadata` | 元数据更新 |

### 3.2 和弦识别技术选型

#### 3.2.1 Omnizart vs BTC-ISMIR19

我们对比测试了两种和弦识别工具：

| 工具 | 支持的和弦类型 | 识别精度 | 处理速度 | 适用场景 | 最终选择 |
|------|--------------|---------|---------|---------|---------|
| **Omnizart** | 25种（12×maj + 12×min + N） | 高 | 快 | 简单流行音乐 | ✅ 默认 |
| **BTC-ISMIR19** | 13种属性（maj, min, dim, aug, sus4, sus2, 7, maj7, m7, dim7, maj6, m6, hdim7） | 高 | 中等 | 复杂音乐，符合论文要求 | ✅ 推荐 |

**选择 Omnizart 的原因**:
1. 识别精度高
2. 处理速度快
3. 输出格式统一
4. 适合简单流行音乐

**选择 BTC-ISMIR19 的原因**:
1. **符合论文要求**: 支持13种和弦类型，与Video2Music论文一致
2. **更丰富的和弦**: 支持7和弦、maj7、dim、sus等复杂和弦
3. **识别精度高**: 基于Transformer架构，识别准确
4. **调性归一化**: 支持自动检测调性并归一化到C/Am

#### 3.2.2 Omnizart 的限制

经过深入研究和测试，我们发现：

**Omnizart 只支持 25 种和弦类型**，这是模型设计上的限制：
- 基于 Harmony Transformer 架构
- 使用 McGill Billboard 数据集训练
- 采用标准的 MIREX 25 类和弦方案
- **无法通过配置参数扩展**到更多和弦类型

**为什么只有 25 种？**
1. **模型设计**: 为了提高稳定性和泛化能力，所有复杂和弦都被映射到最接近的三和弦
2. **数据分布**: 流行音乐中 maj/min 和弦占主导，复杂和弦数量少，类别不平衡
3. **训练数据**: 官方只提供了这套 25 类的预训练模型

**实际测试结果**:
- 测试了多个视频（001.mp4, 054.mp4, 055.mp4）
- 所有视频都只识别出 maj 和 min 两种属性
- 这与 Omnizart 的设计一致，不是错误

#### 3.2.3 BTC-ISMIR19 的优势

**BTC模型特点**:
- 基于双向Transformer + CRF架构
- 支持大词表模式（`--voca True`），可识别多种复杂和弦
- 输出格式：时间戳 + 和弦标签（`.lab`格式）

**处理流程**:
1. **BTC推理**: 调用BTC模型生成原始和弦标签
2. **重采样**: 将变长时间戳转换为每秒一个和弦（1Hz）
3. **类型映射**: 将BTC输出映射到13种标准和弦类型
4. **调性归一化**: 使用music21检测调性，归一化到C大调或A小调

**实际测试结果**:
- 测试视频001.mp4：识别出40种不同的和弦类型
- 包含：maj, min, maj7, m7, 7, dim, dim7, sus2, sus4, maj6, m6等
- 非N和弦占比：94.8%（236/249秒）

**使用建议**:
- **简单音乐**: 使用Omnizart（快速、简单）
- **复杂音乐**: 使用BTC（更丰富、符合论文要求）
- **论文复现**: 使用BTC（与Video2Music论文一致）

### 3.3 模型适配

#### 3.3.1 和弦词汇表调整

**原始设计**: 模型支持 157 种和弦类型
- 12 个根音 × 13 种属性 + 1 个 N = 157 种

**实际数据（Omnizart）**: 25 种和弦类型
- 12 个根音 × 2 种属性（maj/min）+ 1 个 N = 25 种

**实际数据（BTC）**: 动态生成，取决于实际识别结果
- 13 种属性 × 12 个根音 = 多种组合
- 实际识别到的和弦类型：40+种（如 F:maj7, G:7, E:m7, A:min等）

**适配方案**: 
- **Omnizart模式**: 将模型的和弦词汇表从 157 种改为 25 种
- **BTC模式**: 动态生成和弦字典，支持所有识别到的和弦类型

**修改内容**:
1. `utilities/constants.py`: `CHORD_END = 24`（原来是157）
2. `utilities/run_model_vevo.py`: 添加了 `chord_id_to_root_attr()` 函数，支持动态和弦映射
3. `dataset/vevo_dataset.py`: 
   - 更新了 `compute_hits_k_root_attr()` 函数
   - 修复了情感到和弦的映射逻辑（`mapped_tensor` 大小从159改为26）
   - 修复了索引越界问题
   - 添加了属性别名映射（min7→m7, min6→m6, sus→sus4）
   - 添加了容错处理（未知属性使用maj作为默认值）
4. `dataset/batch_prepare_dataset.py`:
   - 自动扫描所有lab文件中的和弦属性
   - 动态更新 `chord_attr.json`，包含所有找到的属性
   - 支持增量更新，保持现有ID不变

**影响**:
- **Omnizart模式**: 模型的 embedding 层和输出层大小从 159 改为 26
- **BTC模式**: 模型的 embedding 层和输出层大小根据实际和弦数量动态调整
- 需要重新初始化模型并重新训练
- 数据集加载和评估代码自动适配

## 4. 技术实现细节

### 4.1 跨环境调用机制

**环境隔离设计**:
- **主环境** (`v2m_train`): 包含 PyTorch、CLIP 等主要依赖
- **Omnizart环境** (`omni`): 独立的 Omnizart 和弦识别环境
- **BTC环境** (`btc`): 独立的 BTC 和弦识别环境

**调用机制**:
```python
# Omnizart调用
subprocess.run([
    OMNI_PYTHON,  # omni环境的Python路径
    script_path,
    ...
], env={**os.environ, 'LD_PRELOAD': FFI_PATH})

# BTC调用
subprocess.run([
    BTC_PYTHON,  # btc环境的Python路径
    btc_test_script,
    ...
], cwd=BTC_REPO_PATH)  # 在BTC仓库目录中运行
```

**环境变量处理**:
- 自动设置 `LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7` 解决 libffi 兼容性
- BTC需要在仓库目录运行以读取 `run_config.yaml`

### 4.2 和弦识别流程

#### 4.2.1 Omnizart流程

**处理步骤**:
1. 调用 Omnizart API 进行和弦识别
2. 读取 CSV 输出文件（`<video_id>.csv`）
3. 解析时间戳和和弦标签
4. 转换为标准格式（`<id>.lab`）

**文件路径处理**:
```python
csv_path = Path(output_dir) / f'{video_id}.csv'
# 支持无扩展名回退
if not csv_path.exists():
    csv_path = Path(output_dir) / f'{video_id}'
```

#### 4.2.2 BTC流程

**处理步骤**:
1. **BTC推理**: 在BTC仓库目录调用 `test.py`，生成原始 `.lab` 文件
2. **解析lab文件**: 读取时间戳和和弦标签
3. **重采样到1Hz**: 每秒一个和弦，选择该秒内占时长最多的和弦
4. **类型映射**: 映射到13种标准和弦类型
5. **调性归一化**（可选）: 使用music21检测调性，归一化到C/Am
6. **输出标准格式**: 生成 `<id>.lab` 文件

**关键实现**:
```python
# 重采样逻辑
def resample_to_1hz(lab_events, total_duration):
    T = int(np.ceil(total_duration))
    result = []
    for sec in range(T):
        dur_by_chord = {}
        for st, et, lab in lab_events:
            inter = max(0.0, min(et, sec+1.0) - max(st, sec))
            if inter > 0:
                dur_by_chord[lab] = dur_by_chord.get(lab, 0.0) + inter
        chord = max(dur_by_chord.items(), key=lambda x: x[1])[0] if dur_by_chord else 'N'
        result.append(chord)
    return result
```

### 4.3 和弦字典管理

**动态生成机制**:
- 自动扫描所有lab文件中的和弦和属性
- 动态更新 `chord.json` 和 `chord_attr.json`
- 支持增量更新，保持现有ID不变

**属性别名映射**:
```python
# 统一属性名称
attr_map = {
    'min7': 'm7',
    'min6': 'm6',
    'sus': 'sus4'
}
```

**容错处理**:
- 未知属性使用 `maj` 作为默认值
- 缺失的根音或属性会记录警告日志

### 4.4 数据集划分管理

**默认模式**:
- 所有视频随机打乱
- 按8:1:1比例分配到训练/验证/测试集

**增量模式** (`--preserve-splits`):
- 读取现有的 `train.txt`, `val.txt`, `test.txt`
- 识别新视频（不在现有划分中的视频）
- 新视频按8:1:1比例分配
- 保持现有划分不变，追加新视频

## 5. 数据格式规范

### 5.1 文件命名规范

- **视频文件**: `<id>.mp4`（如 `001.mp4`, `002.mp4`）
- **输出文件**: 统一使用 `<id>` 作为文件名（不含扩展名）
- **目录结构**: 按特征类型组织到不同子目录

### 5.2 数据格式

**和弦标注格式** (`<id>.lab`):
```
key C major
0 N
1 C:maj
2 G:maj
3 A:min
...
```

**情感标注格式** (`<id>.lab`):
```
time exciting_prob fearful_prob tense_prob sad_prob relaxing_prob neutral_prob
0 0.1234 0.0567 0.0123 0.0456 0.7890 0.0234
1 0.1456 0.0234 0.0345 0.0123 0.8123 0.0123
...
```

**运动特征格式** (`<id>.lab`):
```
0 0.1234
1 0.2345
2 0.3456
...
```

**响度特征格式** (`<id>.lab`):
```
0 0.5678
1 0.6789
2 0.7890
...
```

## 6. 数据集统计信息

### 6.1 基本统计

- **视频总数**: 100 个 MP4 文件
- **已处理视频**: 100 个（✅ 已完成所有特征提取）
- **数据集划分**:
  - **训练集**: 80 个（80%）
  - **验证集**: 10 个（10%）
  - **测试集**: 10 个（10%）
- **增量处理**: 支持使用 `--preserve-splits` 参数增量添加新视频，保持现有划分不变

### 6.2 特征统计

**视觉特征**:
- 语义特征维度: 768（CLIP ViT-L/14）
- 情感类别数: 6（exciting, fearful, tense, sad, relaxing, neutral）
- 平均场景数: 约 10-20 个场景/视频

**音频特征**:
- **和弦类型数**: 109 种（实际识别结果）
- **根音数量**: 12 个（C, C#, D, D#, E, F, F#, G, G#, A, A#, B）
- **属性类型**: 14 种（maj, min, dim, aug, sus4, sus2, 7, maj7, m7, dim7, maj6, m6, hdim7, N）
- **采样率**: 44.1kHz


## 6. 数据集质量评估

### 6.1 优点

1. **特征丰富**: 包含视觉、音频、音乐多模态特征
2. **标注准确**: 使用成熟的工具（Omnizart、CLIP）进行自动标注
3. **格式统一**: 所有特征文件格式一致，便于加载
4. **可扩展性**: 脚本支持增量添加新视频

### 6.2 局限性

1. **音乐风格单一**: 主要是流行音乐，缺少其他风格（爵士、古典、电子等）
2. **依赖外部工具**: 需要配置多个Python环境（v2m_train, omni, btc）
3. **处理时间**: BTC模式处理速度较慢，需要额外的环境配置
4. **数据规模**: 100个视频对于深度学习模型来说规模相对较小，可能需要更多数据

### 6.3 改进方向

1. **增加音乐多样性**: 包含不同风格的音乐（爵士、古典、电子等）
2. **数据增强**: 继续添加新视频，扩大数据集规模
3. **增量处理**: 使用`--preserve-splits`参数，支持增量添加新视频而不打乱现有划分
4. **调性归一化**: 启用BTC的调性归一化功能，统一到C/Am调性
5. **人工标注验证**: 对部分数据进行人工验证，确保标注质量
6. **性能优化**: 优化特征提取流程，提高处理效率

## 7. 代码与工具

### 7.1 主要脚本

1. **`batch_prepare_dataset.py`**: 批量处理脚本（生产环境）
2. **`prepare_dataset.ipynb`**: 交互式处理脚本（调试用）
3. **`convert_chord_format.py`**: 和弦格式转换工具
4. **`test_chord_recognition.py`**: 和弦识别测试脚本

### 7.2 关键配置

- **Omnizart Python**: `/home/jim/anaconda3/envs/omni/bin/python`
- **BTC Python**: `/home/jim/anaconda3/envs/btc/bin/python`（可通过`BTC_PYTHON`环境变量配置）
- **BTC仓库路径**: `/home/jim/BTC-ISMIR19`（可通过`BTC_REPO_PATH`环境变量配置）
- **FFI 库路径**: `/lib/x86_64-linux-gnu/libffi.so.7`
- **CLIP 模型**: `ViT-L/14@336px`
- **场景检测**: `AdaptiveDetector`

### 8.3 使用示例

**基础用法 - 处理所有视频（使用Omnizart，执行所有11个步骤）**:
```bash
cd dataset
python batch_prepare_dataset.py
# 或显式指定
python batch_prepare_dataset.py --chord-method omnizart
```
> **说明**: 会自动执行步骤1→11，处理所有MP4文件

**使用BTC识别和弦（推荐，获得更丰富的和弦类型）**:
```bash
cd dataset
python batch_prepare_dataset.py --chord-method btc
```
> **说明**: 步骤8使用BTC方法，其他步骤相同

**增量处理新视频（保持现有数据集划分）**:
```bash
cd dataset
python batch_prepare_dataset.py --preserve-splits --chord-method btc
```
> **说明**: 新视频按8:1:1分配到训练/验证/测试集，不影响现有划分

**处理单个视频（测试用）**:
```bash
cd dataset
python batch_prepare_dataset.py --video-id 001 --chord-method btc
```
> **说明**: 只处理001.mp4，执行所有11个步骤

**断点续传（跳过已处理的文件）**:
```bash
cd dataset
python batch_prepare_dataset.py --skip-existing --chord-method btc
```
> **说明**: 如果某个视频的某个步骤已完成，会跳过该步骤

**只执行特定步骤（例如：只重新生成和弦）**:
```bash
cd dataset
python batch_prepare_dataset.py \
  --skip-steps frames,motion,semantic,emotion,scene,audio,loudness,midi,note_density,metadata \
  --chord-method btc
```
> **说明**: 只执行步骤8（和弦识别），跳过其他步骤

**处理流程详细说明**:

1. **扫描阶段**: 脚本会自动扫描 `dataset/vevo/` 目录下的所有 `.mp4` 文件

2. **视频处理阶段**: 对每个视频，**严格按照步骤1→11的顺序执行**：
   - **步骤1**: 视频抽帧 → `vevo_frame/`
   - **步骤2**: 运动特征 → `vevo_motion/`
   - **步骤3**: 语义特征 → `vevo_semantic/`
   - **步骤4**: 情感特征 → `vevo_emotion/`
   - **步骤5**: 场景检测 → `vevo_scene/`
   - **步骤6**: 音频提取 → `vevo_audio/`
   - **步骤7**: 响度特征 → `vevo_loudness/`
   - **步骤8**: 和弦识别 → `vevo_chord/`（可选择Omnizart或BTC）
   - **步骤9**: MIDI生成 → `vevo_midi/`
   - **步骤10**: 音符密度 → `vevo_note_density/`
   - **步骤11**: 元数据更新 → `vevo_meta/`（所有视频处理完成后执行）

3. **错误处理**: 
   - 每个步骤都会检查输出文件是否已存在（如果使用 `--skip-existing`）
   - 如果某个步骤失败，会记录错误但继续处理下一个视频
   - 失败的视频不会影响其他视频的处理

4. **元数据更新**: 所有视频处理完成后，自动执行步骤11更新元数据

5. **日志记录**: 处理日志保存在 `batch_prepare_dataset.log` 文件中

## 9. 总结

### 9.1 完成的工作

1. ✅ 收集了 100 个视频文件（MP4格式）
2. ✅ **完成了 100 个视频的特征提取**（所有11个处理步骤）
3. ✅ 实现了自动化批量处理流程
4. ✅ **集成了BTC和弦识别方法**，支持13种和弦类型
5. ✅ **实现了增量处理功能**，支持保持现有数据集划分
6. ✅ 实现了跨环境调用机制（v2m_train, omni, btc）
7. ✅ 将模型适配到动态和弦词汇表（109种和弦类型）
8. ✅ 建立了完整的数据集划分和元数据系统（80/10/10）
9. ✅ **自动更新和弦字典**，识别出109种不同和弦类型
10. ✅ **数据集构建完成**，可用于模型训练

### 9.2 技术贡献

1. **自动化流程**: 实现了端到端的自动化数据集构建流程
2. **跨环境集成**: 解决了不同 Python 环境之间的调用问题（v2m_train, omni, btc）
3. **多方法和弦识别**: 集成Omnizart和BTC两种和弦识别方法，支持不同场景
4. **增量处理**: 实现了数据集增量更新功能，保持现有划分不变
5. **动态和弦字典**: 支持动态生成和更新和弦字典，自动适配新识别到的和弦类型
6. **模型适配**: 成功将模型从 157 种和弦适配到动态和弦词汇表
7. **问题解决**: 解决了多个技术难题（libffi、索引越界、张量大小不匹配、BTC环境配置等）

### 9.3 下一步计划

1. **模型训练**: 使用完整的100个视频数据集训练视频-音乐同步模型
   - 训练集：80个视频
   - 验证集：10个视频
   - 测试集：10个视频
2. **模型评估**: 评估模型性能，分析109种和弦类型的识别效果
3. **数据增强**: 考虑增加更多视频，扩大数据集规模
4. **调性归一化**: 在BTC处理中启用调性归一化功能，统一到C/Am调性
5. **性能优化**: 优化特征提取流程，提高处理效率
6. **文档完善**: 完善数据集文档和使用说明

## 10. 参考文献

1. **Omnizart: A General Toolbox for Automatic Music Transcription**
   - https://www.theoj.org/joss-papers/joss.03391/10.21105.joss.03391.pdf

2. **Omnizart Chord API Documentation**
   - https://music-and-culture-technology-lab.github.io/omnizart-doc/chord/api.html

3. **BTC-ISMIR19: A Bi-Directional Transformer for Musical Chord Recognition**
   - Park et al., ISMIR 2019
   - GitHub: https://github.com/jayg996/BTC-ISMIR19

4. **CLIP: Learning Transferable Visual Representations**
   - Radford et al., 2021

5. **Video2Music: Suitable Music Generation from Videos using an Affective Multimodal Transformer model**
   - arXiv: https://arxiv.org/html/2311.00968v2

6. **Video-Music Synchronization Dataset (MuVi-Sync)**
   - 官方数据集参考

---

**报告生成时间**: 2025-12-04  
**数据集版本**: v1  
**处理脚本版本**: batch_prepare_dataset.py (支持BTC和增量处理)  
**数据集状态**: ✅ 完成（100个视频，109种和弦类型）  
**更新内容**: 
- 集成BTC和弦识别方法（支持13种和弦类型）
- 实现增量处理功能（`--preserve-splits`）
- 自动更新和弦字典（识别出109种不同和弦类型）
- 完成100个视频的特征提取（所有11个处理步骤）

