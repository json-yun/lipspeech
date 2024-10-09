# scripts/model.py

import torch
import torch.nn as nn

class DeepSpeech(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=3):
        super(DeepSpeech, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                            bidirectional=False, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_length, input_size)
        x, _ = self.lstm(x)  # x: (batch, seq_length, hidden_size)
        x = self.fc(x)       # x: (batch, seq_length, num_classes)
        return x
