# scripts/test.py

import torch
from dataset import SpeechDataset, collate_fn
from torch.utils.data import DataLoader
from model import DeepSpeech
import torch.nn.functional as F

def greedy_decoder(output, vocab):
    """
    Greedy decoding for CTC output
    """
    arg_max = torch.argmax(output, dim=2)
    arg_max = arg_max.transpose(0, 1)  # (seq_length, batch)
    decoded = []
    for i in range(arg_max.shape[1]):
        sequence = []
        previous = -1
        for t in range(arg_max.shape[0]):
            current = arg_max[t, i].item()
            if current != previous and current != 0:
                sequence.append(vocab[current])
            previous = current
        decoded.append(''.join(sequence))
    return decoded

def load_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = f.read().strip().split(' ')
    return vocab

def main():
    # 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = 40
    hidden_size = 256
    num_classes = 30
    num_layers = 3
    model_path = 'models/deepspeech_final.pth'
    vocab_path = 'data/vocab.txt'
    
    # Vocab 로드
    vocab = load_vocab(vocab_path)
    
    # 모델 로드
    model = DeepSpeech(input_size, hidden_size, num_classes, num_layers).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    
    # 테스트 데이터 로드
    test_dataset = SpeechDataset('data/preprocessed/test')
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    # 예측 및 디코딩
    for batch in test_loader:
        mfcc, transcript, input_lengths, target_lengths = batch
        mfcc = mfcc.to(device)
        
        with torch.no_grad():
            outputs = model(mfcc)
            log_probs = F.log_softmax(outputs, dim=2)
            log_probs = log_probs.transpose(0, 1)  # (seq_length, batch, num_classes)
            preds = torch.argmax(log_probs, dim=2)
        
        decoded = greedy_decoder(log_probs, vocab)
        print('Predictions:', decoded)

if __name__ == '__main__':
    main()
