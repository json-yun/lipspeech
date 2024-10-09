import os
from tqdm import tqdm
import sys

# call func each file in child path
def iter_dir(dir: str, func: callable) -> None:
    for next in os.scandir(dir):
        if next.is_dir():
            iter_dir(os.path.join(dir, next.name), func)
        else:
            func(os.path.join(dir, next.name))

def prepreprocess(file_name: str):
    if file_name.endswith('.txt'):
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                name, *text = line.split()
                output_file_name = os.path.join(dest_dir, line.split()[0])
                with open(output_file_name + '.txt', 'w', encoding='utf-8') as output:
                    output.write(' '.join(text))

if __name__ == "__main__":
    argv = sys.argv[1:]

    src_dir = argv[0]
    dest_dir = argv[1]
    iter_dir(src_dir, prepreprocess)