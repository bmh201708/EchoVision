# 跨环境使用说明

## 问题描述

`prepare_dataset.ipynb` 使用的 kernel 是 `v2m_clean`，但其中单元格 21（Omnizart 和弦识别）需要使用 `omni` 环境中的 omnizart 库。

## 解决方案

使用 `subprocess` 在 `v2m_clean` kernel 中调用 `omni` 环境的 Python 来运行 omnizart。

### 实现方式

单元格 21 现在会：
1. 创建一个临时的 Python 脚本，包含 omnizart 调用代码
2. 使用 `subprocess.run()` 调用 `/home/jim/anaconda3/envs/omni/bin/python` 执行该脚本
3. 脚本会：
   - 设置 LD_PRELOAD 环境变量（解决 libffi 问题）
   - 导入 omnizart
   - 运行和弦识别
   - 读取生成的 CSV 文件
   - 生成 `.lab` 文件
4. 清理临时脚本文件

### 使用要求

1. **omni 环境必须存在**：确保 `/home/jim/anaconda3/envs/omni/bin/python` 存在
2. **omni 环境中已安装 omnizart**：确保 omnizart 及其依赖已正确安装
3. **checkpoints 已配置**：确保 `~/.omnizart/checkpoints/` 目录中有必要的模型文件

### 验证环境

在运行单元格之前，可以执行以下代码验证：

```python
import os
OMNI_PYTHON = '/home/jim/anaconda3/envs/omni/bin/python'
if os.path.exists(OMNI_PYTHON):
    print(f"✓ omni 环境 Python 存在: {OMNI_PYTHON}")
    import subprocess
    result = subprocess.run([OMNI_PYTHON, '-c', 'import omnizart; print("✓ omnizart 可用")'], 
                          capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("警告:", result.stderr)
else:
    print(f"✗ omni 环境不存在: {OMNI_PYTHON}")
```

### 故障排除

如果遇到问题：

1. **找不到 omni 环境**：
   - 检查 conda 环境列表：`conda env list`
   - 如果环境路径不同，修改单元格中的 `OMNI_PYTHON` 变量

2. **omnizart 导入失败**：
   - 激活 omni 环境：`conda activate omni`
   - 检查 omnizart：`python -c "import omnizart; print(omnizart.__version__)"`
   - 如果失败，重新安装：`pip install omnizart`

3. **libffi 错误**：
   - 确保设置了 LD_PRELOAD（代码中已包含）
   - 检查文件是否存在：`ls -l /lib/x86_64-linux-gnu/libffi.so.7`

4. **CSV 文件未生成**：
   - 检查 omnizart 的输出目录
   - 查看 subprocess 的错误输出（代码中会打印 stderr）

### 替代方案

如果不想使用 subprocess，可以考虑：

1. **将 omnizart 安装到 v2m_clean 环境**：
   ```bash
   conda activate v2m_clean
   pip install omnizart
   ```
   注意：可能会有依赖冲突

2. **切换到 omni kernel**：
   - 在 Jupyter 中切换整个 notebook 的 kernel 到 `omni`
   - 但这样其他单元格可能无法使用 v2m_clean 的依赖

3. **使用独立的 Python 脚本**：
   - 创建一个独立的 `.py` 文件，在 omni 环境中运行
   - 在 notebook 中调用该脚本

当前的 subprocess 方案是最灵活的，允许在同一个 notebook 中使用多个环境。

