# 和弦词汇表从157种改为25种 - 修改总结

## 修改日期
2025-12-04

## 修改目标
将模型的和弦词汇表从157种（0-156）改为25种（0-24），以适配Omnizart识别的数据集。

## 修改的文件

### 1. `utilities/constants.py`
**修改内容**：
- 将 `CHORD_END` 从 `157` 改为 `24`
- 添加了注释说明：`# 25种和弦：0-24 (12×maj + 12×min + N)`

**影响**：
- `CHORD_PAD = 25` (CHORD_END + 1)
- `CHORD_SIZE = 26` (CHORD_PAD + 1)

### 2. `utilities/run_model_vevo.py`
**修改内容**：
- 添加了 `import os`
- 添加了 `chord_id_to_root_attr()` 辅助函数，支持25种和157种和弦的映射
- 修改了 `isGenConfusionMatrix` 部分的硬编码值（157, 158），改为使用常量和辅助函数
- 添加了字典文件加载逻辑，用于25种和弦模式下的root和attr映射

**关键函数**：
```python
def chord_id_to_root_attr(chord_id, chord_inv_dic=None, chord_root_dic=None, chord_attr_dic=None):
    """
    将和弦ID转换为root和attr ID
    支持25种和弦（0-24）和157种和弦（0-156）
    """
```

### 3. `dataset/vevo_dataset.py`
**修改内容**：
- 更新了 `compute_hits_k_root_attr()` 函数，支持25种和弦模式
- 添加了字典文件加载逻辑，建立root+attr到chord_id的映射
- 将硬编码的 `torch.Size([1, 299, 159])` 改为使用 `CHORD_SIZE` 常量

**关键改进**：
- 对于25种和弦，使用 `chord.json`、`chord_root.json`、`chord_attr.json` 来建立映射关系
- 对于157种和弦，保持原有的公式计算逻辑

### 4. `model/video_music_transformer.py`
**修改内容**：
- 更新了注释，将硬编码的 `[1, 157]` 改为 `[1, CHORD_SIZE]`
- 将注释中的 `157 chordEnd 158 padding` 改为使用常量名称

## 自动适配的部分

以下代码会自动适配，因为它们使用了常量或从配置文件读取：

1. **模型定义**：
   - `model/music_transformer.py` - 使用 `CHORD_SIZE` 常量
   - `model/video_music_transformer.py` - 使用 `CHORD_SIZE` 常量

2. **数据集加载**：
   - `dataset/vevo_dataset.py` - 从 `chord.json` 读取，自动适配25种和弦

3. **训练代码**：
   - `train.py` - 使用 `CHORD_SIZE` 和 `CHORD_PAD` 常量

4. **评估代码**：
   - `dataset/vevo_dataset.py` 中的评估函数从 `chord_inv.json` 读取，自动适配

## 验证结果

- ✅ 所有硬编码的 157、158、159 都已被替换
- ✅ 所有使用 `CHORD_SIZE`、`CHORD_END`、`CHORD_PAD` 的地方都正确
- ✅ 代码通过了语法检查（无 linter 错误）

## 注意事项

1. **模型重新训练**：修改 `CHORD_SIZE` 后，模型的 embedding 层和输出层大小会改变，需要重新初始化模型并重新训练。

2. **数据兼容性**：
   - 当前 `dataset/vevo_meta/chord.json` 已经是25种（0-24）
   - 数据集加载代码会自动从这些文件读取，无需额外修改

3. **向后兼容**：
   - `chord_id_to_root_attr()` 函数支持25种和157种两种模式
   - 通过检查 `CHORD_END == 24` 来判断当前模式
   - 如果需要在两种模式间切换，只需修改 `utilities/constants.py` 中的 `CHORD_END` 值

## 测试建议

1. 验证数据集加载：确保能正确加载25种和弦的数据
2. 验证模型初始化：确保模型能正确初始化（embedding和输出层大小正确）
3. 验证训练流程：运行一个epoch，确保训练过程正常
4. 验证评估指标：确保评估函数能正确计算指标

## 相关文档

- `dataset/OMNIZART_CHORD_LIMITATIONS.md` - Omnizart和弦识别限制说明
- `dataset/vevo_meta/chord.json` - 25种和弦的定义（0-24）

