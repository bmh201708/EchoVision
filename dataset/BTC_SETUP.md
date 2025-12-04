# BTC和弦识别环境设置指南

## 概述

BTC（Bi-directional Transformer for Chord Recognition）是Park et al. 2019提出的和弦识别模型，可以识别13种和弦类型，比Omnizart的25种和弦（12×maj + 12×min + N）更符合论文要求。

## 安装步骤

### 1. 克隆BTC-ISMIR19仓库

```bash
cd ~
git clone https://github.com/jayg996/BTC-ISMIR19.git
cd BTC-ISMIR19
```

### 2. 创建Python环境

```bash
# 使用conda创建新环境
conda create -n btc python=3.8
conda activate btc

# 或使用venv
python3.8 -m venv btc_env
source btc_env/bin/activate  # Linux/Mac
# 或
btc_env\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install torch>=1.0
pip install numpy pandas librosa pyrubberband mir_eval pretty_midi pyyaml
pip install music21  # 用于调性归一化
```

**注意**：`pyrubberband` 需要本地安装 Rubber Band Library：

- **Linux (Ubuntu/Debian)**:
  ```bash
  sudo apt-get install rubberband-cli librubberband-dev
  ```

- **macOS**:
  ```bash
  brew install rubberband
  ```

- **Windows**: 需要从源码编译，建议使用WSL或Linux环境

### 4. 下载预训练模型

根据BTC-ISMIR19仓库的README，下载预训练模型权重文件，放在仓库的 `checkpoints/` 目录下。

### 5. 测试BTC安装

```bash
cd BTC-ISMIR19
python test.py --audio_dir /path/to/test/audio --save_dir /path/to/output --voca True
```

如果成功运行并生成 `.lab` 文件，说明安装成功。

## 配置Video2Music项目

### 方法1：使用环境变量（推荐）

```bash
export BTC_PYTHON=/home/jim/anaconda3/envs/btc/bin/python
export BTC_REPO_PATH=/home/jim/BTC-ISMIR19
```

或者使用提供的配置脚本：

```bash
cd /home/jim/Video2Music
source setup_btc_env.sh
```

### 方法2：已自动配置（默认）

`batch_prepare_dataset.py` 中已经配置了默认路径：

```python
# BTC 环境配置（已配置，可通过环境变量覆盖）
BTC_PYTHON = '/home/jim/anaconda3/envs/btc/bin/python'
BTC_REPO_PATH = Path('/home/jim/BTC-ISMIR19')
```

如果需要使用其他路径，可以通过环境变量覆盖。

## 使用方法

### 使用BTC进行和弦识别

```bash
cd /home/jim/Video2Music/dataset
python batch_prepare_dataset.py --chord-method btc
```

### 使用Omnizart（默认）

```bash
python batch_prepare_dataset.py --chord-method omnizart
# 或
python batch_prepare_dataset.py  # 默认使用omnizart
```

### 增量处理时使用BTC

```bash
python batch_prepare_dataset.py --preserve-splits --chord-method btc
```

## BTC识别的13种和弦类型

BTC模型可以识别以下13种和弦类型（quality）：

1. `maj` - 大三和弦
2. `min` - 小三和弦
3. `dim` - 减三和弦
4. `aug` - 增三和弦
5. `sus4` - 挂四和弦
6. `sus2` - 挂二和弦
7. `7` - 属七和弦
8. `maj7` - 大七和弦
9. `m7` - 小七和弦
10. `dim7` - 减七和弦
11. `hdim7` - 半减七和弦
12. `maj6` - 大六和弦
13. `m6` - 小六和弦

加上12个根音（C, C#, D, D#, E, F, F#, G, G#, A, A#, B）和N（无和弦），理论上可以有 12×13+1 = 157种和弦。

## 输出格式

BTC处理后的输出格式与Omnizart相同：

```
key C major
0 C:maj
1 C:maj7
2 F:maj
3 G:7
4 N
...
```

## 注意事项

1. **处理时间**：BTC模型可能比Omnizart慢，特别是对于长音频文件
2. **内存占用**：BTC模型需要更多内存
3. **调性归一化**：如果MIDI文件不存在，会跳过调性归一化步骤
4. **错误处理**：如果BTC识别失败，会自动回退到占位和弦（全N）

## 故障排除

### 问题1：找不到BTC Python

**错误**：`BTC Python 路径未配置或不存在`

**解决**：检查 `BTC_PYTHON` 环境变量或配置文件中的路径是否正确

### 问题2：找不到BTC仓库

**错误**：`BTC 仓库路径未配置或不存在`

**解决**：检查 `BTC_REPO_PATH` 环境变量或配置文件中的路径是否正确

### 问题3：yaml.load() 错误

**错误**：`TypeError: load() missing 1 required positional argument: 'Loader'`

**解决**：这是PyYAML版本兼容性问题，已自动修复。如果遇到，运行：
```bash
cd ~/BTC-ISMIR19
sed -i 's/yaml.load(f)/yaml.load(f, Loader=yaml.FullLoader)/g' utils/hparams.py
```

### 问题4：numpy.float 错误

**错误**：`AttributeError: module 'numpy' has no attribute 'float'`

**解决**：这是NumPy版本兼容性问题，已自动修复。如果遇到，运行：
```bash
cd ~/BTC-ISMIR19
sed -i 's/np\.float/float/g' utils/transformer_modules.py
sed -i 's/np\.int\([^0-9]\)/int\1/g; s/np\.int$/int/g' utils/transformer_modules.py btc_model.py
```

### 问题5：pyrubberband导入失败

**错误**：`ImportError: No module named 'pyrubberband'`

**解决**：安装Rubber Band Library（见上面的安装步骤），或 `pip install pyrubberband`

### 问题6：music21导入失败

**错误**：`ImportError: No module named 'music21'`

**解决**：`pip install music21`，调性归一化功能会被禁用，但不影响基本识别

## 参考

- BTC-ISMIR19 GitHub: https://github.com/jayg996/BTC-ISMIR19
- Paper: "A Bi-Directional Transformer for Musical Chord Recognition" (ISMIR 2019)
- Video2Music Paper: https://arxiv.org/html/2311.00968v2

