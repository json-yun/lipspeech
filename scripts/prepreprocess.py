# python scripts/prepreprocess.py src_dir dest_dir
# example:  python scripts/prepreprocess.py data/dev-clean/ data/train

import os
from tqdm import tqdm
import shutil
import sys
from g2p_en import G2p
import nltk

phonetic_map = {
    'AA': 'a', 'AE': 'b', 'AH': 'c', 'AO': 'd', 'AW': 'e', 'AY': 'f', 'B': 'g', 'CH': 'h', 
    'D': 'i', 'DH': 'j', 'EH': 'k', 'ER': 'l', 'EY': 'm', 'F': 'n', 'G': 'o', 'HH': 'p', 
    'IH': 'q', 'IY': 'r', 'JH': 's', 'K': 't', 'L': 'u', 'M': 'v', 'N': 'w', 'NG': 'x', 
    'OW': 'y', 'OY': 'z', 'P': '!', 'R': '@', 'S': '#', 'SH': '$', 'T': '%', 'TH': '^', 
    'UH': '&', 'UW': '*', 'V': '(', 'W': ')', 'Y': '-', 'Z': '_', 'ZH': '+', ' ': ' ', '\'': ''
}

reversed_phonetic_map = {
    'a': 'AA', 'b': 'AE', 'c': 'AH', 'd': 'AO', 'e': 'AW', 'f': 'AY', 'g': 'B', 'h': 'CH', 
    'i': 'D', 'j': 'DH', 'k': 'EH', 'l': 'ER', 'm': 'EY', 'n': 'F', 'o': 'G', 'p': 'HH',
    'q': 'IH', 'r': 'IY', 's': 'JH', 't': 'K', 'u': 'L', 'v': 'M', 'w': 'N', 'x': 'NG',
    'y': 'OW', 'z': 'OY', '!': 'P', '@': 'R', '#': 'S', '$': 'SH', '%': 'T', '^': 'TH',
    '&': 'UH', '*': 'UW', '(': 'V', ')': 'W', '-': 'Y', '_': 'Z', '+': 'ZH', ' ': ' ', '': ''
}

# call func each file in child path
def iter_dir(dir: str, func: callable) -> None:
    for next in os.scandir(dir):
        if next.is_dir():
            iter_dir(os.path.join(dir, next.name), func)
        else:
            func(os.path.join(dir, next.name))

def encode_phonetic(phonetics: list[str]) -> list[str]:
    return [phonetic_map[w] if not w.endswith(('0', '1', '2')) else phonetic_map[w[:-1]] for w in phonetics]

def decode_phonetic(phonetics_code: str) -> str:
    return ' '.join(reversed_phonetic_map[w] for w in phonetics_code)

def alphabet_to_phonetic(sentence: str) -> list[str]:
    phonetic_encoded = ''.join(encode_phonetic(g2p(sentence)))

    return phonetic_encoded

def prepreprocess(file_name: str):
    if file_name.endswith('.txt'):
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                name, *text = line.split()
                sentence = ' '.join(text)
                encoded = alphabet_to_phonetic(sentence)
                output_file_name = os.path.join(dest_dir, 'transcripts', name)
                with open(output_file_name + '.txt', 'w', encoding='utf-8') as output:
                    output.write(encoded)
    else:
        # shutil.copy(file_name, os.path.join(dest_dir, 'audio'))
        pass

if __name__ == "__main__":
    nltk.download('averaged_perceptron_tagger_eng')
    g2p = G2p()
    argv = sys.argv[1:]

    src_dir = argv[0]
    dest_dir = argv[1]
    iter_dir(src_dir, prepreprocess)