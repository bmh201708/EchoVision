#!/bin/bash
# 运行 generate.py 的启动脚本
# 自动设置 LD_PRELOAD 环境变量以解决 libffi 兼容性问题

export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
cd /home/jim/Video2Music
conda activate v2m_train
python generate.py "$@"


