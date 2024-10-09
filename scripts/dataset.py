# scripts/dataset.py

import os
import torch
from torch.utils.data import Dataset
import librosa

class SpeechDataset(Dataset):
    def __init__(self, preprocessed_dir):
        self.preprocessed_dir = preprocessed_dir
        self.samples = [f for f in os.listdir(preprocessed_dir) if f.endswith('.pt')]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_path = os.path.join(self.preprocessed_dir, self.samples[idx])
        sample = torch.load(sample_path, weights_only=False)
        mfcc = sample['mfcc']
        transcript = sample['transcript']
        return mfcc, transcript

def collate_fn(batch):
    batch_mfcc = [item[0] for item in batch]
    batch_transcript = [item[1] for item in batch]
    
    # 패딩
    mfcc_lengths = [mfcc.shape[0] for mfcc in batch_mfcc]
    transcript_lengths = [transcript.shape[0] for transcript in batch_transcript]
    
    padded_mfcc = torch.nn.utils.rnn.pad_sequence(batch_mfcc, batch_first=True)
    padded_transcript = torch.nn.utils.rnn.pad_sequence(batch_transcript, batch_first=True, padding_value=0)
    
    return padded_mfcc, padded_transcript, torch.tensor(mfcc_lengths), torch.tensor(transcript_lengths)
