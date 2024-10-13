# scripts/preprocess.py
# python scripts/preprocess.py (train/test/val) (--build-vocab)
# example: python scripts/preprocess.py train --build-vocab

import os
import librosa
import numpy as np
import torch
from tqdm import tqdm
import sys

def extract_mfcc(audio_path, n_mfcc=40, sr=16000):
    signal, sr = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    mfcc = mfcc.T  # (Time, n_mfcc)
    return mfcc

def preprocess_data(data_dir, output_dir, vocab_file, n_mfcc=40):
    os.makedirs(output_dir, exist_ok=True)
    audio_dir = os.path.join(data_dir, 'audio')
    transcript_dir = os.path.join(data_dir, 'transcripts')
    
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = f.read().strip().lower().split()
    
    for file_name in tqdm(os.listdir(audio_dir)):
        if file_name.endswith(('.wav', '.pcm', '.flac')):
            if file_name.endswith('.wav'):
                suffix = '.wav'
            elif file_name.endswith('.pcm'):
                suffix = '.pcm'
            elif file_name.endswith('.flac'):
                suffix = '.flac'
            audio_path = os.path.join(audio_dir, file_name)
            transcript_path = os.path.join(transcript_dir, file_name.replace(suffix, '.txt'))
            
            # 특징 추출
            mfcc = extract_mfcc(audio_path, n_mfcc=n_mfcc)
            mfcc = torch.tensor(mfcc, dtype=torch.float)
            
            # 타겟 읽기
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = f.read().strip().lower()
            
            # 타겟을 정수 인덱스로 변환 (간단한 문자 사전 사용)
            transcript_encoded = encode_transcript(transcript, vocab)
            transcript_encoded = torch.tensor(transcript_encoded, dtype=torch.long)
            
            # 저장
            sample_name = file_name.replace(suffix, '.pt')
            torch.save({'mfcc': mfcc, 'transcript': transcript_encoded}, os.path.join(output_dir, sample_name))

def build_vocab(transcripts):
    vocab = set()
    for transcript in transcripts:
        vocab.update(list(transcript))
    vocab = sorted(vocab)
    vocab = ['<blank>'] + vocab  # CTC의 blank 토큰
    return vocab

def encode_transcript(transcript, vocab=None):
    if vocab is None:
        raise ValueError("Vocab is not provided.")
    vocab_dict = {char: idx for idx, char in enumerate(vocab)}
    encoded = [vocab_dict[char] for char in transcript if char in vocab_dict]
    return encoded

if __name__ == '__main__':
    argv = sys.argv[1:]

    # 예제: 데이터 전처리
    preprocess_input_dir = os.path.join('data', argv[0])
    preprocess_output_dir = os.path.join('data/preprocessed/', argv[0])
    vocab_file = 'data/vocab.txt'
    
    # 전체 타겟을 읽어 vocab 생성
    if "--build-vocab" in argv:
        transcripts = []
        transcript_dir = os.path.join(preprocess_input_dir, 'transcripts')
        for file_name in os.listdir(transcript_dir):
            if file_name.endswith('.txt'):
                with open(os.path.join(transcript_dir, file_name), 'r', encoding='utf-8') as f:
                    transcript = f.read().strip().lower()
                    transcripts.append(transcript)

        vocab = build_vocab(transcripts)
    
        # 저장된 vocab을 나중에 사용하기 위해 파일로 저장
        with open(vocab_file, 'w', encoding='utf-8') as f:
            f.write(' '.join(vocab))
    else:
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = f.readline().split()
    
    # 전처리 수행
    preprocess_data(preprocess_input_dir, preprocess_output_dir, vocab_file, n_mfcc=40)
