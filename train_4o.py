import torch
import torch.nn as nn
import torch.optim as optim

class DeepSpeech(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(DeepSpeech, self).__init__()
        # 단방향 LSTM 사용
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=3, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)  # 단방향이므로 hidden_size 그대로 사용

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

### 2. 데이터셋 예제 생성

# 가상 입력 데이터 생성
batch_size = 16
seq_length = 100  # 각 입력 데이터의 길이
input_size = 40  # 입력 벡터의 크기 (예: MFCC 크기)
num_classes = 30  # 음소 또는 문자 개수

# 가상 음성 특징 데이터
inputs = torch.randn(batch_size, seq_length, input_size)

# 가상 타겟 레이블 (정수로 이루어진 레이블 시퀀스)
targets = torch.randint(1, num_classes, (batch_size, seq_length // 2))

# 각 입력과 타겟의 길이 (CTC 손실 함수에 필요)
input_lengths = torch.full((batch_size,), seq_length, dtype=torch.long)
target_lengths = torch.full((batch_size,), seq_length // 2, dtype=torch.long)


### 3. CTC 손실 함수 및 학습 절차
# CTC 손실 함수 정의
ctc_loss = nn.CTCLoss(blank=0)

# 모델, 옵티마이저 정의
model = DeepSpeech(input_size=input_size, hidden_size=256, num_classes=num_classes)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 학습 루프
for epoch in range(5):  # 5번의 에포크 동안 학습
    model.train()
    optimizer.zero_grad()
    
    # 모델을 통해 출력 (로그 확률 계산)
    log_probs = model(inputs).log_softmax(2)
    
    # 손실 계산
    loss = ctc_loss(log_probs.transpose(0, 1), targets, input_lengths, target_lengths)
    loss.backward()
    optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# 학습 후 모델 저장
torch.save(model.state_dict(), 'deepspeech_real_time_model.pth')

### 4. 모델 테스트
# 학습된 모델 로드
model.load_state_dict(torch.load('deepspeech_real_time_model.pth'))
model.eval()

# 테스트 데이터 생성
test_inputs = torch.randn(batch_size, seq_length, input_size)

# 테스트 데이터로 예측
with torch.no_grad():
    test_log_probs = model(test_inputs).log_softmax(2)
    
# 예측된 결과를 greedy decoding (CTC의 출력 결과는 공백과 중복된 문자 제거 필요)
preds = torch.argmax(test_log_probs, dim=2)

print('Predictions:', preds)
