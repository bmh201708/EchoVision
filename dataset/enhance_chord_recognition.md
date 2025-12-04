# 增强和弦识别：从 2 种属性到 12 种属性

## 问题分析

Omnizart 的和弦识别模型**只支持识别 25 种和弦类型**：
- 12 个大三和弦（C:maj, C#:maj, ..., B:maj）
- 12 个小三和弦（C:min, C#:min, ..., B:min）
- 1 个无和弦类别（N）

从"性质（属性）"角度看，它只区分 `maj` 和 `min`，再加一个表示没有和弦的类。

这是 Omnizart 模型设计上的限制，基于标准的 MIREX 25 类和弦方案，无法通过配置参数改变。详见 [`OMNIZART_CHORD_LIMITATIONS.md`](./OMNIZART_CHORD_LIMITATIONS.md)。

## 解决方案

### 方案 1：使用其他和弦识别工具（推荐）

#### 1.1 使用 madmom（功能强大）

```bash
# 安装 madmom
conda activate v2m_train  # 或你的训练环境
pip install madmom
```

**优点**：
- 支持更多和弦类型（maj, min, dim, aug, sus, 7, maj7, min7 等）
- 基于音频特征分析，识别精度较高
- 可以输出更细粒度的和弦信息

**缺点**：
- 需要额外安装依赖
- 处理速度可能较慢

#### 1.2 使用 librosa + music21（组合方案）

```bash
pip install librosa music21
```

**优点**：
- librosa 提供音频特征提取
- music21 提供和弦分析和转换
- 可以自定义和弦识别逻辑

**缺点**：
- 需要自己实现识别逻辑
- 可能需要更多调优

### 方案 2：后处理增强（基于 MIDI 分析）

如果你已经生成了 MIDI 文件，可以基于 MIDI 音符分析推断更复杂的和弦类型：

```python
# 基于 MIDI 音符推断和弦类型
import pretty_midi

def infer_chord_from_midi(midi_path):
    """从 MIDI 文件推断和弦类型"""
    pm = pretty_midi.PrettyMIDI(midi_path)
    
    # 提取音符
    notes = []
    for instrument in pm.instruments:
        for note in instrument.notes:
            notes.append(note.pitch % 12)  # 转换为音级
    
    # 分析音符组合，推断和弦类型
    # 例如：如果有 7 度音，可能是 7 和弦
    # 如果有增 5 度，可能是 aug 和弦
    # ...
```

### 方案 3：混合方案（Omnizart + 后处理）

1. 使用 Omnizart 识别基础和弦（maj/min）
2. 基于音频特征（频谱、谐波等）推断更复杂的和弦类型
3. 使用规则或机器学习模型进行后处理

## 推荐实现：使用 madmom

### 安装和基本使用

```python
from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor

# 初始化处理器
feature_processor = CNNChordFeatureProcessor()
chord_processor = CRFChordRecognitionProcessor()

# 处理音频
features = feature_processor('audio.wav')
chords = chord_processor(features)

# 输出格式：[(start_time, end_time, chord_label), ...]
```

### 集成到 batch_prepare_dataset.py

需要修改 `extract_chord_omnizart` 方法，添加 madmom 作为替代或补充。

## 注意事项

1. **模型兼容性**：如果使用不同的和弦识别工具，需要确保输出格式与官方数据集兼容
2. **识别精度**：更复杂的和弦类型识别精度可能不如简单的 maj/min
3. **处理时间**：更复杂的识别算法可能需要更长的处理时间
4. **数据集一致性**：如果官方数据集也是用 Omnizart 生成的，那么只使用 maj/min 可能更一致

## 建议

**短期方案**：
- 继续使用 Omnizart，只识别 maj/min
- 在训练时，模型会学习到这些基础和弦模式
- 虽然词汇表有 157 种，但实际数据只有 25 种也是可以接受的

**长期方案**：
- 如果需要更丰富的和弦类型，考虑使用 madmom 或其他工具
- 或者使用官方数据集的标注（如果可用）


