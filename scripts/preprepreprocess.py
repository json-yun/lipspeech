from g2p_en import G2p
import nltk

nltk.download('averaged_perceptron_tagger_eng')
g2p = G2p()

sentences = [
    "The knight rode through the night.",
    "She sells sea shells by the sea shore.",
    "ya y a"
]

phonetic_sentences = [" ".join(g2p(sentence)) for sentence in sentences]

for original, phonetic in zip(sentences, phonetic_sentences):
    print(f"Original: {original}")
    print(f"Phonetic: {phonetic}\n")

# "".join(l if l else " " for l in "h a i   l".split(" "))