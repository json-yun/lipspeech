from g2p_en import G2p
import nltk

nltk.download('averaged_perceptron_tagger_eng')
g2p = G2p()

sentences = [
    "The knight rode through the night.",
    "She sells sea shells by the sea shore.",
    "wednesday's happy"
]

phonetic_sentences = [" ".join(g2p(sentence)) for sentence in sentences]

for original, phonetic in zip(sentences, phonetic_sentences):
    print(f"Original: {original}")
    print(f"Phonetic: {phonetic}\n")

# "".join(l if l else " " for l in "h a i   l".split(" "))

# Original: The knight rode through the night.
# Phonetic: DH AH0   N AY1 T   R OW1 D   TH R UW1   DH AH0   N AY1 T   .

# Original: She sells sea shells by the sea shore.
# Phonetic: SH IY1   S EH1 L Z   S IY1   SH EH1 L Z   B AY1   DH AH0   S IY1   SH AO1 R   .

# Original: wednesday's happy
# Phonetic: W EH1 N Z D IY0 Z   HH AE1 P IY0