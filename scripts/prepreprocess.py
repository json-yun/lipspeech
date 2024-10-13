# python scripts/prepreprocess.py src_dir dest_dir
# example:  python scripts/prepreprocess.py data/dev-clean/ data/train
#   arguments : src, dst1, dst2, dst3 and proportion for [dst1, dst2, dst3]
#   usage : python.exe .\script.py src dst1 dst2 dst3 0.6:0.3:0.1 42

import os
from tqdm import tqdm
import shutil
import sys
from g2p_en import G2p
import nltk
import random

phonetic_map = {
    'AA': 'a', 'AE': 'a', 'AH': 'a', 'AO': 'o', 'AW': 'aw', 'AY': 'ay', 'B': 'b', 'CH': 'c',
    'D': 'd', 'DH': '#', 'EH': 'e', 'ER': 'r', 'EY': 'ey', 'F': 'f', 'G': 'g', 'HH': 'h', 'IH': 'i',
    'IY': 'y', 'JH': 'j', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'q', 'OW': 'ow', 'OY': 'oy',
    'P': 'p', 'R': 'r', 'S': 's', 'SH': 'x', 'T': 't', 'TH': '#', 'UH': 'u', 'UW': 'uw', 'V': 'v',
    'W': 'w', 'Y': 'y', 'Z': 'z', 'ZH': 'x', ' ': '_', '\'': '', '': ''
}

reversed_phonetic_map = {
    'a': 'AH', 'o': 'AO', 'aw': 'AW', 'ay': 'AY', 'b': 'B', 'c': 'CH', 'd': 'D', '#': 'TH', 
    'e': 'EH', 'r': 'R', 'ey': 'EY', 'f': 'F', 'g': 'G', 'h': 'HH', 'i': 'IH', 'y': 'Y', 
    'j': 'JH', 'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N', 'q': 'NG', 'ow': 'OW', 'oy': 'OY',
    'p': 'P', 's': 'S', 'x': 'ZH', 't': 'T', 'u': 'UH', 'uw': 'UW', 'v': 'V', 'w': 'W', 
    'z': 'Z', ' ': '', '': ''
    }
def encode_phonetic(phonetics: list[str]) -> list[str]:
    return [phonetic_map[w] if not w.endswith(('0', '1', '2')) else phonetic_map[w[:-1]] for w in phonetics]

def decode_phonetic(phonetics_code: str) -> str:
    return ' '.join(reversed_phonetic_map[w] for w in phonetics_code)

def alphabet_to_phonetic(sentence: str, g2p) -> str:
    phonetic_encoded = ''.join(encode_phonetic(g2p(sentence)))

    return phonetic_encoded

def process_text_files(src_dir: str, g2p) -> dict:
    processed_texts = {}
    for root, _, filenames in os.walk(src_dir):
        for filename in filenames:
            if filename.endswith('.txt'):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f.readlines():
                        name, *text = line.split()
                        sentence = ' '.join(text)
                        encoded = alphabet_to_phonetic(sentence, g2p)
                        processed_texts[name] = encoded
    return processed_texts

def collect_file_pairs(src_dir: str, processed_texts: dict) -> list:
    file_pairs = []
    for root, _, filenames in os.walk(src_dir):
        for filename in filenames:
            if not filename.endswith('.txt'):
                audio_path = os.path.join(root, filename)
                base_name = os.path.splitext(filename)[0]
                if base_name in processed_texts:
                    file_pairs.append((audio_path, base_name, processed_texts[base_name]))
    return file_pairs

def preprocess_file_pair(audio_file: str, text_name: str, encoded_text: str, dest_dir: str):
    # 오디오 파일 처리
    audio_output = os.path.join(dest_dir, 'audio', os.path.basename(audio_file))
    os.makedirs(os.path.dirname(audio_output), exist_ok=True)
    shutil.copy(audio_file, audio_output)
    # 처리된 텍스트 저장
    txt_output = os.path.join(dest_dir, 'transcripts', text_name)
    os.makedirs(os.path.dirname(txt_output), exist_ok=True)
    with open(txt_output + '.txt', 'w', encoding='utf-8') as output:
        output.write(encoded_text)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py src_dir dest_dir1 dest_dir2 dest_dir3")
        sys.exit(1)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    g2p = G2p()
    src_dir = sys.argv[1]
    dest_dirs = sys.argv[2:5]
    # seed = int(sys.argv[6])
    # # 시드 설정
    # random.seed(seed)
    # 텍스트 파일 처리
    print("Processing text files...")
    processed_texts = process_text_files(src_dir, g2p)
    # 파일 쌍 수집
    print("Collecting file pairs...")
    file_pairs = collect_file_pairs(src_dir, processed_texts)
    print(f"Total file pairs found: {len(file_pairs)}")
    # 파일 쌍 목록을 의사 랜덤하게 섞음
    # random.shuffle(file_pairs)
    # 비율에 따라 파일 쌍 분배
    total_pairs = len(file_pairs)
    pair_groups = [[], [], []]
    for i in range(0, total_pairs, 5):
        pair_groups[0].extend(file_pairs[i:i+3])
    for i in range(3, total_pairs, 5):
        pair_groups[1].extend(file_pairs[i:i+1])
    for i in range(4, total_pairs, 5):
        pair_groups[2].extend(file_pairs[i:i+1])
    # 각 그룹의 파일 쌍을 해당 dest_dir로 처리
    for dest_dir, pairs in zip(dest_dirs, pair_groups):
        print(f"Processing {len(pairs)} file pairs for {dest_dir}")
        for audio_file, text_name, encoded_text in tqdm(pairs, desc=f"Processing {dest_dir}"):
            preprocess_file_pair(audio_file, text_name, encoded_text, dest_dir)
    print("Processing completed.")