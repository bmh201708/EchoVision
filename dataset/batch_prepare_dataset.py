#!/usr/bin/env python3
"""
批量处理数据集脚本
从 dataset/vevo/ 目录读取所有 MP4 文件，批量提取特征并生成所需文件。

用法:
    python batch_prepare_dataset.py [选项]

选项:
    --skip-existing: 跳过已存在的输出文件
    --video-id ID: 只处理指定的视频 ID
    --max-workers N: 并行处理的视频数量（默认：1，串行处理）
    --skip-steps STEPS: 跳过指定的处理步骤（用逗号分隔，如：chord,midi）
"""

import os
import sys
import subprocess
import tempfile
import json
import math
import random
import re
import argparse
import logging
from pathlib import Path
from typing import List, Set, Optional
try:
    from tqdm import tqdm
except ImportError:
    # 如果没有 tqdm，使用简单的进度显示
    def tqdm(iterable, desc=None):
        if desc:
            print(desc)
        return iterable

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_prepare_dataset.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 路径配置
DATASET_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DATASET_ROOT.parent
VEVO_DIR = DATASET_ROOT / 'vevo'

# Omnizart 环境配置
OMNI_PYTHON = '/home/jim/anaconda3/envs/omni/bin/python'
FFI_PATH = '/lib/x86_64-linux-gnu/libffi.so.7'

# BTC 环境配置（已配置，可通过环境变量覆盖）
BTC_PYTHON = os.environ.get('BTC_PYTHON', '/home/jim/anaconda3/envs/btc/bin/python')
BTC_REPO_PATH = Path(os.environ.get('BTC_REPO_PATH', '/home/jim/BTC-ISMIR19'))


class DatasetProcessor:
    """数据集处理器"""
    
    def __init__(self, skip_existing: bool = False, skip_steps: Set[str] = None, 
                 preserve_splits: bool = False, chord_method: str = 'omnizart'):
        self.skip_existing = skip_existing
        self.skip_steps = skip_steps or set()
        self.preserve_splits = preserve_splits
        self.chord_method = chord_method.lower()
        self.processed_count = 0
        self.failed_count = 0
        self.failed_videos = []
        
    def check_output_exists(self, video_id: str) -> bool:
        """检查视频的所有输出文件是否已存在"""
        checks = [
            DATASET_ROOT / 'vevo_frame' / video_id,
            DATASET_ROOT / 'vevo_motion' / 'all' / f'{video_id}.lab',
            DATASET_ROOT / 'vevo_semantic' / 'all' / '2d' / 'clip_l14p' / f'{video_id}.npy',
            DATASET_ROOT / 'vevo_emotion' / '6c_l14p' / 'all' / f'{video_id}.lab',
            DATASET_ROOT / 'vevo_scene' / 'all' / f'{video_id}.lab',
            DATASET_ROOT / 'vevo_scene_offset' / 'all' / f'{video_id}.lab',
            DATASET_ROOT / 'vevo_audio' / 'wav' / f'{video_id}.wav',
            DATASET_ROOT / 'vevo_loudness' / 'all' / f'{video_id}.lab',
            DATASET_ROOT / 'vevo_chord' / 'lab_v2_norm' / 'all' / f'{video_id}.lab',
            DATASET_ROOT / 'vevo_midi' / 'all' / f'{video_id}.mid',
            DATASET_ROOT / 'vevo_note_density' / 'all' / f'{video_id}.lab',
        ]
        return all(path.exists() for path in checks)
    
    def extract_frames(self, video_id: str, video_path: Path) -> bool:
        """1. 抽帧"""
        if 'frames' in self.skip_steps:
            return True
            
        try:
            frame_dir = DATASET_ROOT / 'vevo_frame' / video_id
            if self.skip_existing and frame_dir.exists() and list(frame_dir.glob('*.jpg')):
                logger.info(f"  [{video_id}] 跳过抽帧（已存在）")
                return True
                
            frame_dir.mkdir(parents=True, exist_ok=True)
            frame_pattern = frame_dir / f'{video_id}_%03d.jpg'
            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-vf', 'select=bitor(gte(t-prev_selected_t\,1)\,isnan(prev_selected_t))',
                '-vsync', '0', '-qmin', '1', '-q:v', '1', str(frame_pattern),
                '-y'  # 覆盖已存在的文件
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info(f"  [{video_id}] ✓ 抽帧完成")
            return True
        except Exception as e:
            logger.error(f"  [{video_id}] ✗ 抽帧失败: {e}")
            return False
    
    def extract_motion(self, video_id: str, video_path: Path) -> bool:
        """2. 运动特征"""
        if 'motion' in self.skip_steps:
            return True
            
        try:
            import cv2
            motion_dir = DATASET_ROOT / 'vevo_motion' / 'all'
            motion_dir.mkdir(parents=True, exist_ok=True)
            motion_path = motion_dir / f'{video_id}.lab'
            
            if self.skip_existing and motion_path.exists():
                logger.info(f"  [{video_id}] 跳过运动特征（已存在）")
                return True
            
            cap = cv2.VideoCapture(str(video_path))
            motiondict = {0: '0.0000'}
            prev_frame = None
            prev_time = 0.0
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_interval = 1.0
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                curr_time = frame_count / fps
                
                if prev_frame is not None and curr_time - prev_time >= frame_interval:
                    diff = cv2.absdiff(frame, prev_frame)
                    diff_rgb = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
                    motion_value = format(diff_rgb.mean(), '.4f')
                    sec_idx = int(curr_time)
                    motiondict[sec_idx] = str(motion_value)
                    prev_time = curr_time
                
                prev_frame = frame.copy()
                frame_count += 1
            
            cap.release()
            cv2.destroyAllWindows()
            
            # 填充缺失的秒
            max_sec = max(motiondict.keys()) if motiondict else 0
            for i in range(max_sec + 1):
                if i not in motiondict:
                    motiondict[i] = '0.0000'
            
            with open(motion_path, 'w', encoding='utf-8') as f:
                for i in sorted(motiondict.keys()):
                    f.write(f'{i} {motiondict[i]}\n')
            
            logger.info(f"  [{video_id}] ✓ 运动特征完成")
            return True
        except Exception as e:
            logger.error(f"  [{video_id}] ✗ 运动特征失败: {e}")
            return False
    
    def extract_semantic(self, video_id: str) -> bool:
        """3. 语义特征 (CLIP)"""
        if 'semantic' in self.skip_steps:
            return True
            
        try:
            import torch
            import clip
            import numpy as np
            from PIL import Image
            
            sem_dir = DATASET_ROOT / 'vevo_semantic' / 'all' / '2d' / 'clip_l14p'
            sem_dir.mkdir(parents=True, exist_ok=True)
            sem_path = sem_dir / f'{video_id}.npy'
            
            if self.skip_existing and sem_path.exists():
                logger.info(f"  [{video_id}] 跳过语义特征（已存在）")
                return True
            
            frame_dir = DATASET_ROOT / 'vevo_frame' / video_id
            frame_files = sorted(frame_dir.glob('*.jpg'))
            if not frame_files:
                logger.warning(f"  [{video_id}] 没有找到帧文件，跳过语义特征")
                return False
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model, preprocess = clip.load('ViT-L/14@336px', device=device)
            features = torch.zeros((len(frame_files), 768), device=device)
            
            for idx, fpath in enumerate(frame_files):
                image = preprocess(Image.open(fpath)).unsqueeze(0).to(device)
                with torch.no_grad():
                    features[idx] = model.encode_image(image)[0]
            
            np.save(sem_path, features.cpu().numpy())
            logger.info(f"  [{video_id}] ✓ 语义特征完成")
            return True
        except Exception as e:
            logger.error(f"  [{video_id}] ✗ 语义特征失败: {e}")
            return False
    
    def extract_emotion(self, video_id: str) -> bool:
        """4. 情感特征"""
        if 'emotion' in self.skip_steps:
            return True
            
        try:
            import torch
            import clip
            from PIL import Image
            
            emo_dir = DATASET_ROOT / 'vevo_emotion' / '6c_l14p' / 'all'
            emo_dir.mkdir(parents=True, exist_ok=True)
            emo_path = emo_dir / f'{video_id}.lab'
            
            if self.skip_existing and emo_path.exists():
                logger.info(f"  [{video_id}] 跳过情感特征（已存在）")
                return True
            
            frame_dir = DATASET_ROOT / 'vevo_frame' / video_id
            frame_files = sorted(frame_dir.glob('*.jpg'))
            if not frame_files:
                logger.warning(f"  [{video_id}] 没有找到帧文件，跳过情感特征")
                return False
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model, preprocess = clip.load('ViT-L/14@336px', device=device)
            text = clip.tokenize(['exciting', 'fearful', 'tense', 'sad', 'relaxing', 'neutral']).to(device)
            emolist = []
            
            for fpath in frame_files:
                image = preprocess(Image.open(fpath)).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits_per_image, _ = model(image, text)
                    probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
                emolist.append(' '.join([format(p, '.4f') for p in probs]))
            
            with open(emo_path, 'w', encoding='utf-8') as f:
                f.write('time exciting_prob fearful_prob tense_prob sad_prob relaxing_prob neutral_prob\n')
                for i, line in enumerate(emolist):
                    f.write(f'{i} {line}\n')
            
            logger.info(f"  [{video_id}] ✓ 情感特征完成")
            return True
        except Exception as e:
            logger.error(f"  [{video_id}] ✗ 情感特征失败: {e}")
            return False
    
    def extract_scene(self, video_id: str, video_path: Path) -> bool:
        """5. 分镜 + Scene Offset"""
        if 'scene' in self.skip_steps:
            return True
            
        try:
            from scenedetect import open_video, SceneManager
            from scenedetect.detectors import AdaptiveDetector
            
            scene_dir = DATASET_ROOT / 'vevo_scene' / 'all'
            scene_dir.mkdir(parents=True, exist_ok=True)
            scene_offset_dir = DATASET_ROOT / 'vevo_scene_offset' / 'all'
            scene_offset_dir.mkdir(parents=True, exist_ok=True)
            
            scene_path = scene_dir / f'{video_id}.lab'
            scene_offset_path = scene_offset_dir / f'{video_id}.lab'
            
            if self.skip_existing and scene_path.exists() and scene_offset_path.exists():
                logger.info(f"  [{video_id}] 跳过分镜（已存在）")
                return True
            
            video_stream = open_video(str(video_path))
            scene_manager = SceneManager()
            scene_manager.add_detector(AdaptiveDetector())
            scene_manager.detect_scenes(video_stream, show_progress=False)
            scene_list = scene_manager.get_scene_list()
            
            scenedict = {}
            sec = 0
            for idx, scene in enumerate(scene_list):
                end_int = math.ceil(scene[1].get_seconds())
                for s in range(sec, end_int):
                    scenedict[s] = str(idx)
                    sec += 1
            
            with open(scene_path, 'w', encoding='utf-8') as f:
                for i in range(len(scenedict)):
                    f.write(f'{i} {scenedict[i]}\n')
            
            ids = [int(v) for v in scenedict.values()]
            offset_list = []
            if ids:
                current = ids[0]
                offset = 0
                for vid in ids:
                    if vid != current:
                        current = vid
                        offset = 0
                    offset_list.append(offset)
                    offset += 1
            
            with open(scene_offset_path, 'w', encoding='utf-8') as f:
                for i, v in enumerate(offset_list):
                    f.write(f'{i} {v}\n')
            
            logger.info(f"  [{video_id}] ✓ 分镜完成")
            return True
        except Exception as e:
            logger.error(f"  [{video_id}] ✗ 分镜失败: {e}")
            return False
    
    def extract_audio(self, video_id: str, video_path: Path) -> bool:
        """6.1 从 mp4 抽取 wav"""
        if 'audio' in self.skip_steps:
            return True
            
        try:
            audio_dir = DATASET_ROOT / 'vevo_audio' / 'wav'
            audio_dir.mkdir(parents=True, exist_ok=True)
            wav_path = audio_dir / f'{video_id}.wav'
            
            if self.skip_existing and wav_path.exists():
                logger.info(f"  [{video_id}] 跳过音频提取（已存在）")
                return True
            
            cmd = [
                'ffmpeg', '-y', '-i', str(video_path),
                '-vn', '-ac', '1', '-ar', '44100', str(wav_path)
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info(f"  [{video_id}] ✓ 音频提取完成")
            return True
        except Exception as e:
            logger.error(f"  [{video_id}] ✗ 音频提取失败: {e}")
            return False
    
    def extract_loudness(self, video_id: str) -> bool:
        """6.2 响度特征"""
        if 'loudness' in self.skip_steps:
            return True
            
        try:
            import audioop
            import numpy as np
            from pydub import AudioSegment
            from pydub.utils import make_chunks
            
            loudness_dir = DATASET_ROOT / 'vevo_loudness' / 'all'
            loudness_dir.mkdir(parents=True, exist_ok=True)
            loudness_path = loudness_dir / f'{video_id}.lab'
            
            if self.skip_existing and loudness_path.exists():
                logger.info(f"  [{video_id}] 跳过响度特征（已存在）")
                return True
            
            wav_path = DATASET_ROOT / 'vevo_audio' / 'wav' / f'{video_id}.wav'
            if not wav_path.exists():
                logger.warning(f"  [{video_id}] WAV 文件不存在，跳过响度特征")
                return False
            
            audio_data = AudioSegment.from_file(wav_path)
            audio_data = audio_data.set_channels(1).set_frame_rate(44100)
            chunks = make_chunks(audio_data, 1000)  # 每秒
            loudness_per_second = []
            
            for chunk in chunks:
                data = chunk.raw_data
                rms = audioop.rms(data, 2)
                if rms > 0:
                    loudness = 20 * np.log10(rms / 32767)
                    normalized = 10 ** (loudness / 20)
                else:
                    normalized = 0.0
                loudness_per_second.append(format(normalized, '.4f'))
            
            with open(loudness_path, 'w', encoding='utf-8') as f:
                for i, v in enumerate(loudness_per_second):
                    f.write(f'{i} {v}\n')
            
            logger.info(f"  [{video_id}] ✓ 响度特征完成")
            return True
        except Exception as e:
            logger.error(f"  [{video_id}] ✗ 响度特征失败: {e}")
            return False
    
    def extract_chord_omnizart(self, video_id: str) -> bool:
        """6.3 使用 Omnizart 自动和弦识别"""
        if 'chord' in self.skip_steps:
            return True
            
        try:
            chord_dir = DATASET_ROOT / 'vevo_chord' / 'lab_v2_norm' / 'all'
            chord_dir.mkdir(parents=True, exist_ok=True)
            chord_path = chord_dir / f'{video_id}.lab'
            
            if self.skip_existing and chord_path.exists():
                logger.info(f"  [{video_id}] 跳过和弦识别（已存在）")
                return True
            
            wav_path = DATASET_ROOT / 'vevo_audio' / 'wav' / f'{video_id}.wav'
            if not wav_path.exists():
                logger.warning(f"  [{video_id}] WAV 文件不存在，跳过和弦识别")
                return False
            
            if not os.path.exists(OMNI_PYTHON):
                logger.error(f"  [{video_id}] Omnizart Python 不存在: {OMNI_PYTHON}")
                return False
            
            # 设置环境变量
            ffi_path = FFI_PATH
            env = os.environ.copy()
            if os.path.exists(ffi_path):
                if 'LD_PRELOAD' in env:
                    env['LD_PRELOAD'] = ffi_path + ':' + env['LD_PRELOAD']
                else:
                    env['LD_PRELOAD'] = ffi_path
            
            # 创建临时脚本
            script_lines = [
                'import os',
                'import sys',
                'import math',
                'import csv',
                'from pathlib import Path',
                '',
                "# 设置 LD_PRELOAD 以解决 libffi 问题",
                f"ffi_path = '{ffi_path}'",
                'if os.path.exists(ffi_path):',
                "    if 'LD_PRELOAD' in os.environ:",
                "        os.environ['LD_PRELOAD'] = ffi_path + ':' + os.environ['LD_PRELOAD']",
                '    else:',
                "        os.environ['LD_PRELOAD'] = ffi_path",
                '',
                'try:',
                '    from omnizart.chord import app as chord_app',
                '',
                '    wav_path = sys.argv[1]',
                '    output_dir = sys.argv[2]',
                '    video_id = sys.argv[3]',
                '',
                '    # 运行 Omnizart Chord 模型',
                '    midi_result = chord_app.transcribe(wav_path, output=output_dir)',
                '',
                '    # 尝试读取生成的 CSV 文件',
                "    csv_path = Path(output_dir) / f'{video_id}.csv'",
                "    chord_path = Path(output_dir) / f'{video_id}.lab'",
                '',
                "    lines = ['key C major']",
                '',
                '    # 尝试带 .csv 扩展名的文件，如果不存在则尝试无扩展名',
                '    if not csv_path.exists():',
                "        csv_path_no_ext = Path(output_dir) / f'{video_id}'",
                '        if csv_path_no_ext.exists():',
                '            csv_path = csv_path_no_ext',
                '',
                '    if csv_path.exists():',
                '        changes = []',
                '        with open(csv_path, \'r\', encoding=\'utf-8\') as f:',
                '            reader = csv.reader(f)',
                '            next(reader)  # 跳过标题行',
                '            for row in reader:',
                '                if len(row) >= 3:',
                '                    chord_label = row[0] if len(row) > 0 else \'N\'',
                '                    start_time = float(row[1]) if len(row) > 1 else 0.0',
                '                    end_time = float(row[2]) if len(row) > 2 else 0.0',
                '                    changes.append((start_time, end_time, chord_label))',
                '        ',
                '        if changes:',
                '            total_dur = int(math.ceil(changes[-1][1]))',
                '            for sec in range(total_dur):',
                '                label = \'N\'',
                '                for start, end, chord in changes:',
                '                    if start <= sec < end:',
                '                        label = chord or \'N\'',
                '                        break',
                '                lines.append(f\'{sec} {label}\')',
                '        else:',
                '            lines.append(\'0 N\')',
                '    else:',
                '        print(f\'警告: 未找到 CSV 文件，使用占位和弦\')',
                '        lines.append(\'0 N\')',
                '    ',
                '    # 写入结果',
                '    with open(chord_path, \'w\', encoding=\'utf-8\') as f:',
                '        f.write(\'\\n\'.join(lines) + \'\\n\')',
                '',
                '    print(f\'成功生成和弦标注: {chord_path}\')',
                '    sys.exit(0)',
                'except Exception as e:',
                '    print(f\'Omnizart 失败: {e}\')',
                '    import traceback',
                '    traceback.print_exc()',
                '    sys.exit(1)',
            ]
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write('\n'.join(script_lines))
                temp_script = f.name
            
            try:
                result = subprocess.run(
                    [OMNI_PYTHON, temp_script, str(wav_path), str(chord_dir), video_id],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if result.returncode == 0:
                    logger.info(f"  [{video_id}] ✓ 和弦识别完成")
                    return True
                else:
                    logger.warning(f"  [{video_id}] Omnizart 失败，使用占位和弦")
                    # 生成占位和弦
                    from pydub import AudioSegment
                    from math import ceil
                    audio = AudioSegment.from_file(wav_path)
                    dur_sec = ceil(len(audio) / 1000)
                    with open(chord_path, 'w', encoding='utf-8') as f:
                        f.write('key C major\n')
                        for t in range(dur_sec):
                            f.write(f'{t} N\n')
                    logger.info(f"  [{video_id}] ✓ 已生成占位和弦")
                    return True
            finally:
                if os.path.exists(temp_script):
                    os.unlink(temp_script)
        except Exception as e:
            logger.error(f"  [{video_id}] ✗ 和弦识别失败: {e}")
            return False
    
    def extract_chord_btc(self, video_id: str) -> bool:
        """6.3 使用 BTC 模型进行和弦识别（13种和弦类型）"""
        if 'chord' in self.skip_steps:
            return True
            
        try:
            # 尝试相对导入，如果失败则尝试绝对导入
            try:
                from .btc_chord_processor import process_btc_chords
            except ImportError:
                from btc_chord_processor import process_btc_chords
            
            chord_dir = DATASET_ROOT / 'vevo_chord' / 'lab_v2_norm' / 'all'
            chord_dir.mkdir(parents=True, exist_ok=True)
            chord_path = chord_dir / f'{video_id}.lab'
            
            if self.skip_existing and chord_path.exists():
                logger.info(f"  [{video_id}] 跳过BTC和弦识别（已存在）")
                return True
            
            wav_path = DATASET_ROOT / 'vevo_audio' / 'wav' / f'{video_id}.wav'
            if not wav_path.exists():
                logger.warning(f"  [{video_id}] WAV 文件不存在，跳过BTC和弦识别")
                return False
            
            if not os.path.exists(BTC_PYTHON) or BTC_PYTHON == '/path/to/btc/env/bin/python':
                logger.error(f"  [{video_id}] BTC Python 路径未配置或不存在: {BTC_PYTHON}")
                logger.error(f"  请设置环境变量 BTC_PYTHON 和 BTC_REPO_PATH，或修改 batch_prepare_dataset.py 中的配置")
                return False
            
            if not BTC_REPO_PATH.exists() or str(BTC_REPO_PATH) == '/path/to/BTC-ISMIR19':
                logger.error(f"  [{video_id}] BTC 仓库路径未配置或不存在: {BTC_REPO_PATH}")
                logger.error(f"  请设置环境变量 BTC_REPO_PATH，或修改 batch_prepare_dataset.py 中的配置")
                return False
            
            # 检查是否有MIDI文件用于调性归一化
            midi_path = DATASET_ROOT / 'vevo_midi' / 'all' / f'{video_id}.mid'
            midi_path = midi_path if midi_path.exists() else None
            
            success, output_path = process_btc_chords(
                wav_path=wav_path,
                output_dir=chord_dir,
                video_id=video_id,
                btc_python=BTC_PYTHON,
                btc_repo_path=BTC_REPO_PATH,
                midi_path=midi_path,
                normalize_key=True
            )
            
            if success:
                logger.info(f"  [{video_id}] ✓ BTC和弦识别完成（13种和弦类型）")
                return True
            else:
                logger.warning(f"  [{video_id}] BTC识别失败，使用占位和弦")
                # 生成占位和弦
                from pydub import AudioSegment
                from math import ceil
                audio = AudioSegment.from_file(wav_path)
                dur_sec = ceil(len(audio) / 1000)
                with open(chord_path, 'w', encoding='utf-8') as f:
                    f.write('key C major\n')
                    for t in range(dur_sec):
                        f.write(f'{t} N\n')
                logger.info(f"  [{video_id}] ✓ 已生成占位和弦")
                return True
                
        except ImportError as e:
            logger.error(f"  [{video_id}] ✗ BTC处理器导入失败: {e}")
            logger.error(f"  请确保 btc_chord_processor.py 文件存在")
            return False
        except Exception as e:
            logger.error(f"  [{video_id}] ✗ BTC和弦识别失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def generate_midi(self, video_id: str) -> bool:
        """6.4 生成 MIDI"""
        if 'midi' in self.skip_steps:
            return True
            
        try:
            import types
            import subprocess as sp
            
            # 设置环境变量
            ffi_path = '/usr/lib/x86_64-linux-gnu/libffi.so.7'
            if os.path.exists(ffi_path):
                os.environ['LD_PRELOAD'] = ffi_path + (':' + os.environ.get('LD_PRELOAD', ''))
            
            # 卸载 fluidsynth，避免冲突
            sp.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'fluidsynth'], 
                   check=False, stdout=sp.PIPE, stderr=sp.PIPE)
            os.environ.setdefault('PRETTY_MIDI_USE_FLUIDSYNTH', '0')
            
            def safe_import_pretty_midi():
                try:
                    import pretty_midi as pm
                    return pm
                except OSError:
                    sys.modules['fluidsynth'] = types.ModuleType('fluidsynth')
                    return __import__('pretty_midi')
            
            pretty_midi = safe_import_pretty_midi()
            
            midi_dir = DATASET_ROOT / 'vevo_midi' / 'all'
            midi_dir.mkdir(parents=True, exist_ok=True)
            chord_dir = DATASET_ROOT / 'vevo_chord' / 'lab_v2_norm' / 'all'
            mid_path = midi_dir / f'{video_id}.mid'
            chord_path = chord_dir / f'{video_id}.lab'
            
            if self.skip_existing and mid_path.exists():
                logger.info(f"  [{video_id}] 跳过 MIDI 生成（已存在）")
                return True
            
            if not chord_path.exists():
                logger.warning(f"  [{video_id}] 和弦标注不存在，跳过 MIDI 生成")
                return False
            
            def chord_to_intervals(chord_text: str) -> List[int]:
                text = chord_text.lower()
                if 'dim' in text or 'o' in text:
                    return [0, 3, 6]
                if 'aug' in text or '+' in text:
                    return [0, 4, 8]
                if 'sus2' in text:
                    return [0, 2, 7]
                if 'sus' in text:
                    return [0, 5, 7]
                if 'min6' in text:
                    return [0, 3, 7, 9]
                if 'min7' in text or (':7' in text and 'min' in text):
                    return [0, 3, 7, 10]
                if '7' in text and 'maj' in text:
                    return [0, 4, 7, 11]
                if '7' in text:
                    return [0, 4, 7, 10]
                if 'min' in text or 'm:' in text or text.endswith('m'):
                    return [0, 3, 7]
                return [0, 4, 7]
            
            def parse_root(chord_text: str) -> str:
                m = re.match(r'([A-Ga-g][b#]?)', chord_text)
                return m.group(1).upper() if m else 'C'
            
            pm = pretty_midi.PrettyMIDI()
            instrument = pretty_midi.Instrument(program=0)
            
            with open(chord_path, encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2 or parts[0] == 'key':
                        continue
                    try:
                        t = int(parts[0])
                    except ValueError:
                        continue
                    chord = parts[1]
                    if chord == 'N':
                        continue
                    root = parse_root(chord)
                    intervals = chord_to_intervals(chord)
                    try:
                        root_pitch = pretty_midi.note_name_to_number(root + '4')
                    except Exception:
                        root_pitch = pretty_midi.note_name_to_number('C4')
                    velocity = 80
                    duration = 1.0
                    for iv in intervals:
                        n = pretty_midi.Note(velocity=velocity, pitch=root_pitch + iv, start=t, end=t+duration)
                        instrument.notes.append(n)
            
            pm.instruments.append(instrument)
            pm.write(str(mid_path))
            logger.info(f"  [{video_id}] ✓ MIDI 生成完成")
            return True
        except Exception as e:
            logger.error(f"  [{video_id}] ✗ MIDI 生成失败: {e}")
            return False
    
    def extract_note_density(self, video_id: str) -> bool:
        """6.5 Note Density"""
        if 'note_density' in self.skip_steps:
            return True
            
        try:
            import types
            import subprocess as sp
            
            # 设置环境变量
            ffi_path = '/usr/lib/x86_64-linux-gnu/libffi.so.7'
            if os.path.exists(ffi_path):
                os.environ['LD_PRELOAD'] = ffi_path + (':' + os.environ.get('LD_PRELOAD', ''))
            
            sp.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'fluidsynth'], 
                   check=False, stdout=sp.PIPE, stderr=sp.PIPE)
            os.environ.setdefault('PRETTY_MIDI_USE_FLUIDSYNTH', '0')
            
            def safe_import_pretty_midi():
                try:
                    import pretty_midi as pm
                    return pm
                except OSError:
                    sys.modules['fluidsynth'] = types.ModuleType('fluidsynth')
                    return __import__('pretty_midi')
            
            pretty_midi = safe_import_pretty_midi()
            
            midi_dir = DATASET_ROOT / 'vevo_midi' / 'all'
            chord_dir = DATASET_ROOT / 'vevo_chord' / 'lab_v2_norm' / 'all'
            note_density_dir = DATASET_ROOT / 'vevo_note_density' / 'all'
            note_density_dir.mkdir(parents=True, exist_ok=True)
            
            mid_path = midi_dir / f'{video_id}.mid'
            chord_path = chord_dir / f'{video_id}.lab'
            note_density_path = note_density_dir / f'{video_id}.lab'
            
            if self.skip_existing and note_density_path.exists():
                logger.info(f"  [{video_id}] 跳过 Note Density（已存在）")
                return True
            
            if not mid_path.exists() or not chord_path.exists():
                logger.warning(f"  [{video_id}] MIDI 或和弦标注不存在，跳过 Note Density")
                return False
            
            # 计算和弦标注长度
            ct = 0
            with open(chord_path, encoding='utf-8') as f:
                for line in f:
                    line_arr = line.strip().split(' ')
                    if len(line_arr) > 1:
                        ct += 1
            
            midi_data = pretty_midi.PrettyMIDI(str(mid_path))
            total_time = int(midi_data.get_end_time())
            note_density_list = []
            
            for i in range(total_time + 1):
                start_time, end_time = i, i + 1
                total_notes = 0
                for instrument in midi_data.instruments:
                    for note in instrument.notes:
                        if note.start < end_time and note.end > start_time:
                            total_notes += 1
                note_density_list.append(total_notes / float(end_time - start_time))
            
            with open(note_density_path, 'w', encoding='utf-8') as f:
                for i in range(ct - 1):
                    if i < len(note_density_list):
                        f.write(f'{i} {note_density_list[i]}\n')
                    else:
                        f.write(f'{i} 0\n')
            
            logger.info(f"  [{video_id}] ✓ Note Density 完成")
            return True
        except Exception as e:
            logger.error(f"  [{video_id}] ✗ Note Density 失败: {e}")
            return False
    
    def update_metadata(self, video_ids: List[str]) -> bool:
        """更新元数据（在所有视频处理完成后）"""
        if 'metadata' in self.skip_steps:
            return True
            
        try:
            meta_root = DATASET_ROOT / 'vevo_meta'
            meta_root.mkdir(parents=True, exist_ok=True)
            split_root = meta_root / 'split' / 'v1'
            split_root.mkdir(parents=True, exist_ok=True)
            chord_root = DATASET_ROOT / 'vevo_chord' / 'lab_v2_norm' / 'all'
            
            labs = list(chord_root.glob('*.lab'))
            
            # 扫描和弦集合和属性集合
            chords = set()
            attrs_found = set()
            ids = []
            for lab in labs:
                ids.append(lab.stem)
                with open(lab, encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2 and parts[0] != 'key':
                            chord = parts[1]
                            chords.add(chord)
                            # 提取属性
                            if ':' in chord:
                                attr = chord.split(':')[1]
                                attrs_found.add(attr)
                            elif chord != 'N':
                                # 如果没有冒号，可能是根音（如'C'），属性为maj
                                attrs_found.add('maj')
            
            ids = sorted(ids)
            chords = sorted(chords)
            
            # 生成 chord 词典（增量更新：保持现有ID，新和弦追加）
            existing_chord2id = {}
            if (meta_root / 'chord.json').exists():
                try:
                    existing_chord2id = json.loads((meta_root / 'chord.json').read_text(encoding='utf-8'))
                except Exception as e:
                    logger.warning(f"读取现有 chord.json 失败，将重新生成: {e}")
                    existing_chord2id = {}
            
            # 识别新和弦
            new_chords = set(chords) - set(existing_chord2id.keys())
            
            # 构建新的映射：保持现有ID，新和弦追加
            chord2id = existing_chord2id.copy()
            if existing_chord2id:
                next_id = max(existing_chord2id.values()) + 1
            else:
                next_id = 0
            
            for chord in sorted(new_chords):
                chord2id[chord] = next_id
                next_id += 1
            
            # 如果完全没有现有映射，按原逻辑处理
            if not existing_chord2id:
                chord2id = {c: i for i, c in enumerate(chords)}
            
            (meta_root / 'chord.json').write_text(
                json.dumps(chord2id, ensure_ascii=False, indent=2), encoding='utf-8')
            (meta_root / 'chord_inv.json').write_text(
                json.dumps({v: k for k, v in chord2id.items()}, ensure_ascii=False, indent=2), encoding='utf-8')
            
            # 根/属性词典
            roots = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'N']
            
            # 基础13种和弦类型（论文要求）
            base_attrs = ['maj', 'min', 'dim', 'aug', 'sus4', 'sus2', '7', 'maj7', 'm7', 'dim7', 'maj6', 'm6', 'hdim7', 'N']
            # 添加找到的所有属性（包括变体）
            all_attrs = list(base_attrs)
            for attr in sorted(attrs_found):
                # 处理属性别名映射
                attr_normalized = attr
                if attr == 'min7' or attr == 'm7':
                    attr_normalized = 'm7'
                elif attr == 'min6' or attr == 'm6':
                    attr_normalized = 'm6'
                elif attr == 'sus':
                    attr_normalized = 'sus4'
                elif attr not in all_attrs:
                    all_attrs.append(attr_normalized)
            
            # 确保N在最后
            if 'N' in all_attrs:
                all_attrs.remove('N')
            all_attrs.append('N')
            
            root2id = {r: i for i, r in enumerate(roots)}
            attr2id = {a: i for i, a in enumerate(all_attrs)}
            
            (meta_root / 'chord_root.json').write_text(
                json.dumps(root2id, ensure_ascii=False, indent=2), encoding='utf-8')
            (meta_root / 'chord_root_inv.json').write_text(
                json.dumps({v: k for k, v in root2id.items()}, ensure_ascii=False, indent=2), encoding='utf-8')
            (meta_root / 'chord_attr.json').write_text(
                json.dumps(attr2id, ensure_ascii=False, indent=2), encoding='utf-8')
            (meta_root / 'chord_attr_inv.json').write_text(
                json.dumps({v: k for k, v in attr2id.items()}, ensure_ascii=False, indent=2), encoding='utf-8')
            
            # 保存 top_chord 列表
            (meta_root / 'top_chord.txt').write_text('\n'.join(chords) + '\n', encoding='utf-8')
            
            # 写 idlist
            (meta_root / 'idlist.txt').write_text('\n'.join(ids) + '\n', encoding='utf-8')
            
            # 数据集划分：保持现有划分或重新划分
            if self.preserve_splits:
                # 读取现有划分
                existing_train = set()
                existing_val = set()
                existing_test = set()
                
                if (split_root / 'train.txt').exists():
                    train_content = (split_root / 'train.txt').read_text(encoding='utf-8').strip()
                    if train_content:
                        existing_train = set(line.strip() for line in train_content.split('\n') if line.strip())
                
                if (split_root / 'val.txt').exists():
                    val_content = (split_root / 'val.txt').read_text(encoding='utf-8').strip()
                    if val_content:
                        existing_val = set(line.strip() for line in val_content.split('\n') if line.strip())
                
                if (split_root / 'test.txt').exists():
                    test_content = (split_root / 'test.txt').read_text(encoding='utf-8').strip()
                    if test_content:
                        existing_test = set(line.strip() for line in test_content.split('\n') if line.strip())
                
                # 识别新视频
                all_existing = existing_train | existing_val | existing_test
                new_ids = [vid for vid in ids if vid not in all_existing]
                
                # 新视频按 8:1:1 分配
                if new_ids:
                    random.shuffle(new_ids)
                    n_new = len(new_ids)
                    new_train = new_ids[:int(0.8 * n_new)]
                    new_val = new_ids[int(0.8 * n_new):int(0.9 * n_new)]
                    new_test = new_ids[int(0.9 * n_new):]
                    
                    # 合并
                    train = sorted(list(existing_train) + new_train)
                    val = sorted(list(existing_val) + new_val)
                    test = sorted(list(existing_test) + new_test)
                    
                    logger.info(f"保持现有划分: {len(existing_train)}/{len(existing_val)}/{len(existing_test)} 个视频")
                    logger.info(f"新增视频: {len(new_train)}/{len(new_val)}/{len(new_test)} 个视频")
                else:
                    train = sorted(list(existing_train))
                    val = sorted(list(existing_val))
                    test = sorted(list(existing_test))
                    logger.info(f"没有新视频，保持现有划分不变")
            else:
                # 原有逻辑：重新随机划分所有视频
                ids_shuf = ids.copy()
                random.shuffle(ids_shuf)
                n = len(ids_shuf)
                train, val, test = (
                    ids_shuf[:int(0.8 * n)],
                    ids_shuf[int(0.8 * n):int(0.9 * n)],
                    ids_shuf[int(0.9 * n):]
                )
                logger.info(f"重新划分数据集: {len(train)}/{len(val)}/{len(test)} 个视频")
            
            for name, split in [('train', train), ('val', val), ('test', test)]:
                (split_root / f'{name}.txt').write_text('\n'.join(split) + '\n', encoding='utf-8')
            
            logger.info(f"✓ 元数据更新完成: {len(chords)} 个和弦, {len(ids)} 个视频")
            return True
        except Exception as e:
            logger.error(f"✗ 元数据更新失败: {e}")
            return False
    
    def process_video(self, video_id: str, video_path: Path) -> bool:
        """处理单个视频"""
        logger.info(f"处理视频: {video_id}")
        
        steps = [
            ('抽帧', self.extract_frames),
            ('运动特征', self.extract_motion),
            ('语义特征', self.extract_semantic),
            ('情感特征', self.extract_emotion),
            ('分镜', self.extract_scene),
            ('音频提取', self.extract_audio),
            ('响度特征', self.extract_loudness),
            ('和弦识别', self.extract_chord_omnizart if self.chord_method == 'omnizart' else self.extract_chord_btc),
            ('MIDI 生成', self.generate_midi),
            ('Note Density', self.extract_note_density),
        ]
        
        success = True
        for step_name, step_func in steps:
            if step_name.lower().replace(' ', '_') in self.skip_steps:
                continue
            try:
                if step_name in ['抽帧', '运动特征', '分镜', '音频提取']:
                    result = step_func(video_id, video_path)
                else:
                    result = step_func(video_id)
                if not result:
                    success = False
                    logger.warning(f"  [{video_id}] 步骤 '{step_name}' 失败")
            except Exception as e:
                logger.error(f"  [{video_id}] 步骤 '{step_name}' 异常: {e}")
                success = False
        
        if success:
            self.processed_count += 1
            logger.info(f"✓ [{video_id}] 处理完成")
        else:
            self.failed_count += 1
            self.failed_videos.append(video_id)
            logger.error(f"✗ [{video_id}] 处理失败")
        
        return success
    
    def process_all(self, video_ids: Optional[List[str]] = None):
        """处理所有视频"""
        if video_ids is None:
            # 扫描所有 MP4 文件
            video_files = sorted(VEVO_DIR.glob('*.mp4'))
            video_ids = [f.stem for f in video_files]
        
        if not video_ids:
            logger.warning("没有找到要处理的视频文件")
            return
        
        logger.info(f"找到 {len(video_ids)} 个视频文件")
        
        # 处理每个视频
        for video_id in tqdm(video_ids, desc="处理视频"):
            video_path = VEVO_DIR / f'{video_id}.mp4'
            if not video_path.exists():
                logger.warning(f"视频文件不存在: {video_path}")
                continue
            
            if self.skip_existing and self.check_output_exists(video_id):
                logger.info(f"跳过 {video_id}（所有输出文件已存在）")
                continue
            
            self.process_video(video_id, video_path)
        
        # 更新元数据（除非明确跳过）
        if 'metadata' not in self.skip_steps:
            logger.info("更新元数据...")
            processed_ids = [vid for vid in video_ids if vid not in self.failed_videos]
            self.update_metadata(processed_ids)
        else:
            logger.info("跳过元数据更新（已指定 --skip-steps metadata）")
        
        # 输出统计
        logger.info("=" * 60)
        logger.info(f"处理完成!")
        logger.info(f"  成功: {self.processed_count}")
        logger.info(f"  失败: {self.failed_count}")
        if self.failed_videos:
            logger.info(f"  失败的视频: {', '.join(self.failed_videos)}")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='批量处理视频数据集')
    parser.add_argument('--skip-existing', action='store_true',
                       help='跳过已存在的输出文件')
    parser.add_argument('--video-id', type=str,
                       help='只处理指定的视频 ID')
    parser.add_argument('--skip-steps', type=str, default='',
                       help='跳过指定的处理步骤（用逗号分隔，如：chord,midi）')
    parser.add_argument('--max-workers', type=int, default=1,
                       help='并行处理的视频数量（默认：1，串行处理）')
    parser.add_argument('--preserve-splits', action='store_true',
                       help='保持现有的训练/验证/测试集划分，新视频按8:1:1比例分配')
    parser.add_argument('--chord-method', type=str, default='omnizart',
                       choices=['omnizart', 'btc'],
                       help='和弦识别方法：omnizart（25种和弦）或 btc（13种和弦类型）')
    
    args = parser.parse_args()
    
    skip_steps = set(s.lower().strip() for s in args.skip_steps.split(',') if s.strip())
    
    processor = DatasetProcessor(
        skip_existing=args.skip_existing,
        skip_steps=skip_steps,
        preserve_splits=args.preserve_splits,
        chord_method=args.chord_method
    )
    
    video_ids = [args.video_id] if args.video_id else None
    
    processor.process_all(video_ids)


if __name__ == '__main__':
    main()

