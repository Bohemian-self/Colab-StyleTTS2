import os
import sys
import subprocess
import shutil
import random
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
from faster_whisper import WhisperModel
import torch
import re

# Import from 'Slicer2.py'
class Slicer:
    def __init__(self,
                 sr: int,
                 threshold: float = -40.,
                 min_length: int = 5000,
                 min_interval: int = 300,
                 hop_size: int = 20,
                 max_sil_kept: int = 5000):
        if not min_length >= min_interval >= hop_size:
            raise ValueError('The following condition must be satisfied: min_length >= min_interval >= hop_size')
        if not max_sil_kept >= hop_size:
            raise ValueError('The following condition must be satisfied: max_sil_kept >= hop_size')
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        if len(waveform.shape) > 1:
            return waveform[:, begin * self.hop_size: min(waveform.shape[1], end * self.hop_size)]
        else:
            return waveform[begin * self.hop_size: min(waveform.shape[0], end * self.hop_size)]

    def slice(self, waveform):
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform
        if (samples.shape[0] + self.hop_size - 1) // self.hop_size <= self.min_length:
            return [waveform]
        
        # Get RMS Values by Librosa
        rms_list = librosa.feature.rms(
            y=samples, 
            frame_length=self.win_size, 
            hop_length=self.hop_size
        ).squeeze(0)
        
        sil_tags = []
        silence_start = None
        clip_start = 0
        for i, rms in enumerate(rms_list):
            if rms < self.threshold:
                if silence_start is None:
                    silence_start = i
                continue
            if silence_start is None:
                continue
                
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = i - silence_start >= self.min_interval and i - clip_start >= self.min_length
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
                
            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start: i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[i - self.max_sil_kept: silence_start + self.max_sil_kept + 1].argmin()
                pos += i - self.max_sil_kept
                pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None
            
        total_frames = rms_list.shape[0]
        if silence_start is not None and total_frames - silence_start >= self.min_interval:
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start: silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))
            
        if len(sil_tags) == 0:
            return [waveform]
        else:
            chunks = []
            if sil_tags[0][0] > 0:
                chunks.append(self._apply_slice(waveform, 0, sil_tags[0][0]))
            for i in range(len(sil_tags) - 1):
                chunks.append(self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0]))
            if sil_tags[-1][1] < total_frames:
                chunks.append(self._apply_slice(waveform, sil_tags[-1][1], total_frames))
            return chunks

class AudioProcessor:
    def __init__(self, input_file, output_base="raw"):
        self.input_file = Path(input_file)
        self.base_name = self.input_file.stem
        self.output_base = Path(output_base)
        self.raw_dir = self.output_base / self.base_name
        self.wavs_dir = self.raw_dir / "wavs"
        
        # Create Dictorinary
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.wavs_dir.mkdir(parents=True, exist_ok=True)
        
    def run_demucs(self):
        """使用demucs分离人声"""
        print("🎵 步骤1: 使用Demucs分离人声...")
        try:
            cmd = [
                "demucs", 
                "-n", "htdemucs", 
                "--out", "separated", 
                str(self.input_file)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            print("✅ Demucs处理完成")
        except subprocess.CalledProcessError as e:
            print(f"❌ Demucs处理失败: {e}")
            raise
        except FileNotFoundError:
            print("❌ 未找到demucs命令，请确保已安装demucs")
            raise
    
    def resample_audio(self):
        """使用ffmpeg重采样音频"""
        print("🎵 步骤2: 重采样音频到单声道24kHz...")
        vocals_path = Path(f"separated/htdemucs/{self.base_name}/vocals.wav")
        output_path = self.raw_dir / f"{self.base_name}.wav"
        
        if not vocals_path.exists():
            raise FileNotFoundError(f"未找到分离后的人声文件: {vocals_path}")
        
        try:
            cmd = [
                "ffmpeg", "-i", str(vocals_path),
                "-ac", "1", "-ar", "24000",
                "-y", str(output_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            print("✅ 音频重采样完成")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"❌ 音频重采样失败: {e}")
            raise
    
    def slice_audio(self, audio_path):
        """使用librosa切片音频"""
        print("🎵 步骤3: 切片音频为2-10秒片段...")
        try:
            # Loading audios
            audio, sr = librosa.load(audio_path, sr=24000, mono=True)
            
            # Create Slicer
            slicer = Slicer(
                sr=sr,
                threshold=-40,
                min_length=2000,  # 2秒
                min_interval=300,
                hop_size=10,
                max_sil_kept=500
            )
            
            # Slice
            chunks = slicer.slice(audio)
            
            # Saving Slices
            saved_files = []
            for i, chunk in enumerate(tqdm(chunks, desc="保存音频切片")):
                if len(chunk.shape) > 1:
                    chunk = chunk.T
                
                output_file = self.wavs_dir / f"{self.base_name}_{i:03d}.wav"
                sf.write(output_file, chunk, sr)
                saved_files.append(output_file.name)
            
            print(f"✅ 音频切片完成，共生成 {len(saved_files)} 个片段")
            return saved_files
            
        except Exception as e:
            print(f"❌ 音频切片失败: {e}")
            raise
    
    def transcribe_audio(self):
        """使用Faster-Whisper转录音频"""
        print("🎵 步骤4: 使用Whisper Large-v3转录音频...")
        try:
            # Loading Whisper Model
            model = WhisperModel("large-v3", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")
            
            # Search for all audios
            audio_files = sorted(self.wavs_dir.glob("*.wav"))
            
            transcripts = []
            for audio_file in tqdm(audio_files, desc="转录音频"):
                segments, info = model.transcribe(
                    str(audio_file),
                    language=None,  # Detect Language by itself
                    beam_size=5
                )
                
                text = " ".join([segment.text for segment in segments]).strip()
                # Cleaning texts
                text = re.sub(r'\s+', ' ', text)
                transcripts.append((audio_file.name, text))
            
            # Saving the transcript result
            transcript_file = self.raw_dir / "transcribed_texts.txt"
            with open(transcript_file, 'w', encoding='utf-8') as f:
                for filename, text in transcripts:
                    f.write(f"{filename}|{text}\n")
            
            print("✅ 音频转录完成")
            return transcripts
            
        except Exception as e:
            print(f"❌ 音频转录失败: {e}")
            raise
    
    def phonemize_text(self, transcripts):
        """音素化文本"""
        print("🎵 步骤5: 音素化文本...")
        try:
            # Using espeak-ng to phonemize
            phonemized_data = []
            
            for filename, text in tqdm(transcripts, desc="音素化文本"):
                if not text.strip():
                    phonemized_text = ""
                else:
                    # transform texts into phonemize
                    try:
                        cmd = ["espeak", "-q", "--phonout=-", "-v", "en", text]
                        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                        phonemized_text = result.stdout.strip()
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        # 如果espeak不可用，使用简单的替换规则
                        phonemized_text = self._simple_phonemize(text)
                
                phonemized_data.append((filename, phonemized_text))
            
            # Saving phonemized result into OOD_texts.txt and train_list.txt
            OOD_texts_file = self.raw_dir / "OOD_texts.txt"
            with open(OOD_texts_file, 'w', encoding='utf-8') as f:
                for filename, phonemized_text in phonemized_data:
                    f.write(f"{filename}|{phonemized_text}|0\n")


            train_list_file = self.raw_dir / "train_list.txt"
            with open(train_list_file, 'w', encoding='utf-8') as f:
                for filename, phonemized_text in phonemized_data:
                    f.write(f"{filename}|{phonemized_text}|0\n")
            
            print("✅ 文本音素化完成")
            return phonemized_data
            
        except Exception as e:
            print(f"❌ 文本音素化失败: {e}")
            raise
    
    def _simple_phonemize(self, text):
        """简单的音素化替代方案"""
        # 这是一个简化的音素映射表，实际使用时建议使用专业的音素化库
        phoneme_map = {
            'a': 'æ', 'e': 'ɛ', 'i': 'ɪ', 'o': 'ɒ', 'u': 'ʌ',
            'th': 'θ', 'sh': 'ʃ', 'ch': 'tʃ', 'ng': 'ŋ'
        }
        
        words = text.lower().split()
        phonemized_words = []
        for word in words:
            phonemized_word = word
            for key, value in phoneme_map.items():
                phonemized_word = phonemized_word.replace(key, value)
            phonemized_words.append(phonemized_word)
        
        return ' '.join(phonemized_words)
    
    def split_train_val(self):
        """划分训练集和验证集"""
        print("🎵 步骤6: 划分训练集和验证集...")
        try:
            train_list_file = self.raw_dir / "train_list.txt"
            
            if not train_list_file.exists():
                raise FileNotFoundError(f"未找到训练列表文件: {train_list_file}")
            
            # 读取所有行
            with open(train_list_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 随机打乱
            random.shuffle(lines)
            
            # 按1:10比例划分验证集
            val_ratio = 1 / 10
            val_size = max(1, int(len(lines) * val_ratio))
            
            val_lines = lines[:val_size]
            train_lines = lines[val_size:]
            
            # 保存验证集
            val_list_file = self.raw_dir / "val_list.txt"
            with open(val_list_file, 'w', encoding='utf-8') as f:
                f.writelines(val_lines)
            
            # 更新训练集（移除验证集部分）
            with open(train_list_file, 'w', encoding='utf-8') as f:
                f.writelines(train_lines)
            
            print(f"✅ 数据集划分完成 - 训练集: {len(train_lines)} 条, 验证集: {len(val_lines)} 条")
            
        except Exception as e:
            print(f"❌ 数据集划分失败: {e}")
            raise
    
    def process(self):
        """执行完整的处理流程"""
        try:
            print(f"🚀 开始处理音频文件: {self.input_file}")
            
            # 步骤1: Demucs分离
            self.run_demucs()
            
            # 步骤2: 重采样
            resampled_audio = self.resample_audio()
            
            # 步骤3: 切片
            sliced_files = self.slice_audio(resampled_audio)
            
            # 步骤4: 转录
            transcripts = self.transcribe_audio()
            
            # 步骤5: 音素化
            self.phonemize_text(transcripts)
            
            # 步骤6: 划分数据集
            self.split_train_val()
            
            print(f"🎉 {self.base_name} Process Successfully Finished!")
            print(f"📁 输出目录: {self.raw_dir}")
            
        except Exception as e:
            print(f"💥 处理失败: {e}")
            raise

def main():
    if len(sys.argv) != 2:
        print("用法: python audio_processor.py <input_wav_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"错误: 文件 {input_file} 不存在")
        sys.exit(1)
    
    processor = AudioProcessor(input_file)
    processor.process()

if __name__ == "__main__":
    main()
