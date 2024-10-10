
1. **폴더 구조 생성**:
   ```bash
   mkdir -p data/train/audio data/train/transcripts
   mkdir -p data/val/audio data/val/transcripts
   mkdir -p data/test/audio data/test/transcripts
   mkdir scripts models
   ```

2. **필요한 스크립트 생성**:
   - `preprocess.py`
   - `dataset.py`
   - `model.py`
   - `train.py`
   - `test.py`
   - `create_example_data.py`

3. **예제 데이터 생성**:
   ```bash
   python scripts/create_example_data.py
   ```

4. **데이터 전처리**:
   ```bash
   python scripts/preprocess.py
   ```

5. **모델 학습**:
   ```bash
   python scripts/train.py
   ```

6. **모델 테스트**:
   ```bash
   python scripts/test.py
   ```
