# Omnizart 和弦识别限制说明

## 核心结论

**Omnizart 自带的和弦模型一共只有 25 个类别**：

- 👉 12 个大三和弦（C:maj, C#:maj, D:maj, D#:maj, E:maj, F:maj, F#:maj, G:maj, G#:maj, A:maj, A#:maj, B:maj）
- 👉 12 个小三和弦（C:min, C#:min, D:min, D#:min, E:min, F:min, F#:min, G:min, G#:min, A:min, A#:min, B:min）
- 👉 1 个 "无和弦 / 无和声" 类别（N / no-chord）

**从"性质（属性）"角度看，它只区分 `maj` 和 `min`，再加一个表示没有和弦的类。**

所以只看到 `maj` 和 `min` 是**完全正常、符合设计的**，不是安装或使用错误。

## 技术细节

### 1. 模型设计依据

Omnizart 的和弦识别基于 **Harmony Transformer**，并使用 **McGill Billboard 数据集** + 标准的 **MIREX 25 类和弦方案**（12 大、12 小、1 无和弦）。

参考论文：
- [Omnizart: A General Toolbox for Automatic Music Transcription](https://www.theoj.org/joss-papers/joss.03391/10.21105.joss.03391.pdf)
- [Omnizart Chord API 文档](https://music-and-culture-technology-lab.github.io/omnizart-doc/chord/api.html)

### 2. 为什么只有 25 类？

#### 2.1 模型设计就是"小词表"

为了提高稳定性和泛化能力，Omnizart 把所有复杂和弦都映射到"最接近的三和弦"：
- `Cmaj7` → `C:maj`
- `Dm7` → `D:min`
- `Cdim` → `C:min`（或根据上下文映射）
- `Caug` → `C:maj`（或根据上下文映射）

#### 2.2 真实数据里复杂和弦太少、类别极度不平衡

在流行音乐中：
- 出现频率最高的是大三和弦和小三和弦
- 七和弦、增减和弦、各种扩展和弦数量少得多
- 如果直接把所有复杂和弦都变成单独的类别，训练时会面临严重类别不平衡问题
- 效果往往比只做 maj/min 还差

许多自动和弦识别系统（包括其他库，如 autochord）也采取同样的 25 类方案。

#### 2.3 官方预训练模型目前只有这一套词表

Omnizart 的文档和 JOSS 论文都写明，当前只提供这套 25 类的和弦识别模型。配置文件里可以调的是模型结构、训练参数等，但**不能简单通过修改配置就让它突然支持 maj7 / 9 / dim 等新类别**，那需要重新标注数据 + 重新训练模型。

## 实际输出格式

Omnizart 输出的 CSV 文件格式示例：

```csv
Chord,Start,End
C:maj,0.000,2.300
G:min,2.300,4.600
N,4.600,5.000
A:maj,5.000,7.200
...
```

**你永远不会看到 `maj7`, `min7`, `dim`, `aug`, `sus4` 这些更复杂的标记**——因为**模型本就没学这些类别**。

## 与项目模型的关系

### 项目模型支持 157 种和弦类型

项目的 `chord.json` 定义了 157 种和弦类型：
- 12 个根音（C, C#, D, D#, E, F, F#, G, G#, A, A#, B）
- 13 种属性（N, 无属性即maj, dim, sus4, min7, min, sus2, aug, dim7, maj6, hdim7, 7, min6, maj7）
- 总计：12 × 13 + 1 = 157 种

### 数据集实际使用的和弦类型

虽然模型支持 157 种，但实际数据集中：
- **Omnizart 识别结果**：只有 25 种（12×maj + 12×min + N）
- **官方数据集标注**：主要也是 maj/min，偶尔有 7、maj7 等复杂和弦

这是**完全正常的**：
1. 模型词汇表大是为了支持更丰富的和弦类型（如果数据中有）
2. 实际数据集中简单和弦占主导是音乐风格的特点
3. 训练时，模型会学习到这些基础和弦模式

## 如果你想要更多和弦类型，有什么选择？

### 方案 1：自己重训练更大词表的模型（最硬核）

- 基于 Harmony Transformer / Omnizart 的 chord 模块，改 label 编码 & 词表，让它学习 maj7、7、dim、aug 等
- 需要：
  - 一个有丰富和弦标注（含七和弦、扩展和弦等）的数据集
  - 修改标签解析 & 训练代码
  - 重新训练模型
- 这基本是研究级工作，如果只是应用层使用，会比较重

### 方案 2：用其他支持大词表的和弦识别库 / 工具

- 一些库（如 `crema` 的 chord 模型）支持大规模的和弦词表（上百种和弦）
- 参考：[crema 文档](https://crema.readthedocs.io/en/latest/models.html)
- 需要自己对接音频、调用模型、再把结果整合进你的工作流

### 方案 3：结合规则后处理做"细化"

- 利用 Omnizart 的 maj/min 结果 + 你自己从音高 / 和声音色中提取的信息，做人为规则判断（例如听第四个音是否构成 7 或 maj7）
- 这属于"半自动"方案，可靠性和工作量就看你有多想折腾了

### 方案 4：使用官方数据集的标注（推荐）

- 如果官方数据集有更丰富的和弦标注，直接使用
- 虽然标注中偶尔有复杂和弦，但大部分还是 `maj/min`
- 这是最简单、最可靠的方案

## 总结

### 回答两个关键问题

1. **Omnizart 能识别出几种和弦类型？**
   - **25 类**：12×`maj` + 12×`min` + 1×`no-chord`

2. **为什么我只看到 `maj` 和 `min` 两种属性？**
   - 因为模型的"性质词表"里**只有 maj、min 和 no-chord**
   - 所有复杂和弦在训练和推理时都被"压缩"成这些基本三和弦
   - 这是 Omnizart 设计上的取舍，并不是你的环境有问题

### 建议

**短期方案**：
- ✅ 继续使用 Omnizart，只识别 maj/min
- ✅ 在训练时，模型会学习到这些基础和弦模式
- ✅ 虽然词汇表有 157 种，但实际数据只有 25 种也是可以接受的

**长期方案**：
- 如果需要更丰富的和弦类型，考虑使用官方数据集的标注
- 或者使用其他支持大词表的和弦识别工具（如 crema）
- 或者自己训练模型（如果有标注数据）

## 参考资料

1. [Omnizart JOSS 论文](https://www.theoj.org/joss-papers/joss.03391/10.21105.joss.03391.pdf)
2. [Omnizart Chord API 文档](https://music-and-culture-technology-lab.github.io/omnizart-doc/chord/api.html)
3. [crema 模型文档](https://crema.readthedocs.io/en/latest/models.html)
4. [autochord GitHub](https://github.com/cjbayron/autochord)


