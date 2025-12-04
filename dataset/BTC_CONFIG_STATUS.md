# BTC环境配置状态

## 配置完成情况

### ✅ 已完成

1. **BTC-ISMIR19仓库**: 已克隆到 `/home/jim/BTC-ISMIR19`
2. **Conda环境**: 已创建 `btc` 环境（Python 3.8）
3. **Python依赖**: 已安装
   - ✅ torch 2.4.1+cu121
   - ✅ numpy, pandas, librosa
   - ✅ mir_eval, pretty_midi
   - ✅ pyyaml, music21
4. **预训练模型**: 已存在
   - ✅ `test/btc_model_large_voca.pt` (12MB)
   - ✅ `test/btc_model.pt` (12MB)
5. **配置文件**: 已更新
   - ✅ `batch_prepare_dataset.py` 中已配置默认路径
   - ✅ 创建了 `setup_btc_env.sh` 配置脚本

### ⚠️ 待处理

1. **Rubber Band Library**: 系统库未安装
   - 需要: `sudo apt-get install rubberband-cli librubberband-dev`
   - 影响: `pyrubberband` 可能无法正常工作（但BTC基本功能不受影响）

## 快速使用

### 设置环境变量（可选）

```bash
source /home/jim/Video2Music/setup_btc_env.sh
```

### 测试BTC识别

```bash
cd /home/jim/Video2Music/dataset
python batch_prepare_dataset.py --chord-method btc --video-id 001
```

### 使用Omnizart（默认）

```bash
python batch_prepare_dataset.py --chord-method omnizart --video-id 001
```

## 验证安装

运行以下命令验证BTC环境：

```bash
cd ~/BTC-ISMIR19
conda activate btc
python test.py --audio_dir test --save_dir test_output --voca True
```

如果成功生成 `.lab` 文件，说明安装正确。

## 路径配置

- **BTC Python**: `/home/jim/anaconda3/envs/btc/bin/python`
- **BTC仓库**: `/home/jim/BTC-ISMIR19`
- **预训练模型**: `~/BTC-ISMIR19/test/btc_model_large_voca.pt`

