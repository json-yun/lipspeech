# scripts/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from dataset import SpeechDataset, collate_fn
from torch.utils.data import DataLoader
from model import DeepSpeech
from tqdm import tqdm

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        mfcc, transcript, input_lengths, target_lengths = batch
        mfcc, transcript = mfcc.to(device), transcript.to(device)
        input_lengths, target_lengths = input_lengths.to(device), target_lengths.to(device)
        
        optimizer.zero_grad()
        outputs = model(mfcc)  # (batch, seq_length, num_classes)
        log_probs = outputs.log_softmax(2)  # CTC expects log probabilities
        
        # CTC 손실은 (T, N, C) 형태를 기대
        log_probs = log_probs.transpose(0, 1)  # (seq_length, batch, num_classes)
        
        loss = criterion(log_probs, transcript, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def evaluate(model, device, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            mfcc, transcript, input_lengths, target_lengths = batch
            mfcc, transcript = mfcc.to(device), transcript.to(device)
            input_lengths, target_lengths = input_lengths.to(device), target_lengths.to(device)
            
            outputs = model(mfcc)
            log_probs = outputs.log_softmax(2)
            log_probs = log_probs.transpose(0, 1)
            
            loss = criterion(log_probs, transcript, input_lengths, target_lengths)
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    return avg_loss

def main():
    # 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = 40  # MFCC 수
    hidden_size = 256
    num_classes = 42  # vocab 크기
    num_layers = 3
    num_epochs = 20
    learning_rate = 1e-3
    
    # 데이터 로드
    train_dataset = SpeechDataset('data/preprocessed/train')
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    
    val_dataset = SpeechDataset('data/preprocessed/val')
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    # 모델, 손실 함수, 옵티마이저
    model = DeepSpeech(input_size, hidden_size, num_classes, num_layers).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 학습 루프
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, criterion)
        val_loss = evaluate(model, device, val_loader, criterion)
        print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 모델 저장
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f'models/deepspeech_epoch_{epoch}.pth')

    # 최종 모델 저장
    torch.save(model.state_dict(), 'models/deepspeech_final.pth')

if __name__ == '__main__':
    main()
