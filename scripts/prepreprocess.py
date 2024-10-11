# python scripts/prepreprocess.py src_dir dest_dir
# example:  python scripts/prepreprocess.py data/dev-clean/ data/train

import os
from tqdm import tqdm
import shutil
import sys
from g2p_en import G2p
import nltk

phonetic_map = {
    'AA': 'a', 'AE': 'a', 'AH': 'a', 'AO': 'o', 'AW': 'aw', 'AY': 'ay', 'B': 'b', 'CH': 'c',
    'D': 'd', 'DH': '#', 'EH': 'e', 'ER': 'r', 'EY': 'ey', 'F': 'f', 'G': 'g', 'HH': 'h', 'IH': 'i',
    'IY': 'y', 'JH': 'j', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'q', 'OW': 'ow', 'OY': 'oy',
    'P': 'p', 'R': 'r', 'S': 's', 'SH': 'x', 'T': 't', 'TH': '#', 'UH': 'u', 'UW': 'uw', 'V': 'v',
    'W': 'w', 'Y': 'y', 'Z': 'z', 'ZH': 'x', ' ': ' ', '\'': '', '': ''
}

reversed_phonetic_map = {
    'a': 'AH', 'o': 'AO', 'aw': 'AW', 'ay': 'AY', 'b': 'B', 'c': 'CH', 'd': 'D', '#': 'TH', 
    'e': 'EH', 'r': 'R', 'ey': 'EY', 'f': 'F', 'g': 'G', 'h': 'HH', 'i': 'IH', 'y': 'Y', 
    'j': 'JH', 'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N', 'q': 'NG', 'ow': 'OW', 'oy': 'OY',
    'p': 'P', 's': 'S', 'x': 'ZH', 't': 'T', 'u': 'UH', 'uw': 'UW', 'v': 'V', 'w': 'W', 
    'z': 'Z', ' ': ' ', '': ''
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
        shutil.copy(file_name, os.path.join(dest_dir, 'audio'))
        pass

if __name__ == "__main__":
    nltk.download('averaged_perceptron_tagger_eng')
    g2p = G2p()
    argv = sys.argv[1:]

    src_dir = argv[0]
    dest_dir = argv[1]
    iter_dir(src_dir, prepreprocess)