# 和弦词汇表更新到157种 - 修改说明

## 修改日期
2025-01-XX

## 修改目标
将和弦词汇表从153种补齐到157种（12根音×13属性 + N），以支持完整的157种和弦训练。

## 修改的文件

### 1. `dataset/vevo_meta/chord.json`
**修改内容**：
- 补齐了4种缺失的和弦：
  - `A#:dim7` → ID 153
  - `D:dim7` → ID 154
  - `G:dim` → ID 155
  - `G:dim7` → ID 156
- 现在包含完整的157种和弦（ID范围：0-156）

### 2. `dataset/vevo_meta/chord_inv.json`
**修改内容**：
- 自动更新了反向映射
- 确保与`chord.json`完全一致

### 3. `utilities/constants.py`
**修改内容**：
```python
# 修改前
CHORD_END = 24  # 25种和弦：0-24
CHORD_PAD = 25
CHORD_SIZE = 26

# 修改后
CHORD_END = 156  # 157种和弦：0-156 (12根音×13属性 + N)
CHORD_PAD = 157
CHORD_SIZE = 158
```

## 157种和弦的构成

### 12个根音
C, C#, D, D#, E, F, F#, G, G#, A, A#, B

### 13种属性
maj, min, dim, aug, sus4, sus2, 7, maj7, m7, dim7, maj6, m6, hdim7

### 组合
- 12根音 × 13属性 = 156种和弦组合
- 加上N（无和弦）= 157种和弦
- ID范围：0-156

## 注意事项

### 1. 模型需要重新训练
**重要**：由于`CHORD_SIZE`从26变为158，模型的embedding层和输出层大小发生了变化：
- Embedding层：`nn.Embedding(CHORD_SIZE, d_model)` 从26维变为158维
- 输出层：`nn.Linear(d_model, CHORD_SIZE)` 从26维变为158维

**影响**：
- ❌ **旧的模型权重无法直接使用**（维度不匹配）
- ✅ **需要重新训练模型**
- ✅ 训练数据可以继续使用（因为数据中的和弦ID仍然有效）

### 2. 数据兼容性
**好消息**：
- ✅ 现有的训练数据（lab文件）仍然有效
- ✅ 数据中的和弦ID（0-152）仍然有效
- ✅ 新增的4种和弦（ID 153-156）在数据中可能不存在，但不影响训练

**说明**：
- 如果训练数据中没有这4种和弦，模型仍然可以训练
- 这4种和弦的embedding会随机初始化，但不会影响其他和弦的学习
- 如果未来数据中出现这4种和弦，模型已经准备好了

### 3. 训练建议

**开始训练前**：
1. ✅ 确认`CHORD_END = 156`已更新
2. ✅ 确认`chord.json`包含157种和弦
3. ✅ 确认`chord_inv.json`已更新
4. ⚠️ 删除旧的模型权重（如果存在）
5. ⚠️ 重新初始化模型

**训练时**：
- 模型会自动处理未出现的和弦（ID 153-156）
- 这些和弦的embedding会保持随机初始化状态
- 不影响其他153种和弦的学习

**如果数据中出现新和弦**：
- 如果数据中出现ID 153-156的和弦，模型会正常学习
- 如果数据中出现ID > 156的和弦，会报错（需要更新`CHORD_END`）

## 验证步骤

### 1. 验证文件完整性
```bash
python3 -c "
import json
from pathlib import Path

chord_file = Path('dataset/vevo_meta/chord.json')
with open(chord_file) as f:
    chord_dic = json.load(f)

print(f'和弦总数: {len(chord_dic)}')
print(f'最大ID: {max(chord_dic.values())}')
print(f'最小ID: {min(chord_dic.values())}')

# 验证是否包含所有157种和弦
roots = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
attrs = ['maj', 'min', 'dim', 'aug', 'sus4', 'sus2', '7', 'maj7', 'm7', 'dim7', 'maj6', 'm6', 'hdim7']
expected = set()
for root in roots:
    for attr in attrs:
        expected.add(f'{root}:{attr}')
expected.add('N')

actual = set(chord_dic.keys())
if expected == actual:
    print('✓ 包含所有157种和弦！')
else:
    print(f'缺少: {sorted(expected - actual)}')
    print(f'多余: {sorted(actual - expected)}')
"
```

### 2. 验证常量配置
```bash
python3 -c "
from utilities.constants import CHORD_END, CHORD_PAD, CHORD_SIZE
print(f'CHORD_END: {CHORD_END} (应该是156)')
print(f'CHORD_PAD: {CHORD_PAD} (应该是157)')
print(f'CHORD_SIZE: {CHORD_SIZE} (应该是158)')
"
```

## 常见问题

### Q1: 补齐的4种和弦在数据中不存在，会影响训练吗？
**A**: 不会。这4种和弦的embedding会随机初始化，但不会影响其他和弦的学习。如果未来数据中出现这些和弦，模型已经准备好了。

### Q2: 需要重新处理数据吗？
**A**: 不需要。现有的训练数据仍然有效，因为数据中的和弦ID（0-152）仍然有效。

### Q3: 旧的模型权重还能用吗？
**A**: 不能。由于`CHORD_SIZE`从26变为158，模型的embedding层和输出层大小发生了变化，需要重新训练。

### Q4: 如果数据中出现ID > 156的和弦怎么办？
**A**: 会报错。需要更新`CHORD_END`和`chord.json`，然后重新训练。

### Q5: 为什么补齐到157种而不是保持153种？
**A**: 157种是完整的理论组合（12根音×13属性 + N），补齐后可以：
- 支持所有可能的和弦组合
- 避免未来数据中出现新和弦时的错误
- 与论文中的157种和弦方案一致

## 总结

✅ **已完成**：
- 补齐了4种缺失的和弦
- 更新了`chord.json`和`chord_inv.json`
- 更新了`constants.py`中的`CHORD_END`

⚠️ **需要注意**：
- 需要重新训练模型（旧的权重无法使用）
- 训练数据仍然有效，不需要重新处理

✅ **可以开始训练**：
- 配置已更新完成
- 可以按照157种和弦开始训练

---

**文档生成时间**：2025-01-XX  
**项目版本**：Video2Music v1.0  
**作者**：项目组

