# scripts/create_example_data.py

import os
import numpy as np
import librosa
import soundfile as sf

def create_sine_wave(freq, duration, sr=16000):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    y = 0.5*np.sin(2*np.pi*freq*t)
    return y

def save_wav(y, sr, path):
    sf.write(path, y, sr)

def main():
    os.makedirs('data/train/audio', exist_ok=True)
    os.makedirs('data/train/transcripts', exist_ok=True)
    
    # 예제 데이터 생성
    samples = [
        {'id': '0001', 'text': 'hello'},
        {'id': '0002', 'text': 'world'},
        {'id': '0003', 'text': 'test'},
        {'id': '0004', 'text': 'speech recognition'},
        {'id': '0005', 'text': 'deep learning'}
    ]
    
    for sample in samples:
        wav_path = f"data/train/audio/{sample['id']}.wav"
        txt_path = f"data/train/transcripts/{sample['id']}.txt"
        
        # 음성 생성 (여기서는 단순히 다양한 주파수의 sine wave를 생성)
        freq = 300 + int(sample['id']) * 100  # 각 샘플마다 다른 주파수
        y = create_sine_wave(freq, duration=1.0)  # 1초 길이
        save_wav(y, 16000, wav_path)
        
        # 타겟 저장
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(sample['text'].lower())

if __name__ == '__main__':
    main()
