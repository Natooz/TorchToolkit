
from torch import randint, randn

from flexformer import Transformer
from flexformer.sampling import beam_search


if __name__ == '__main__':
    seq2seq = Transformer(2, 2, 256, 4, 512, num_classes=200)
    encoder = Transformer(2, 0, 256, 4, 512, num_classes=200)
    decoder = Transformer(0, 2, 256, 4, 512, num_classes=200)

    src = randint(200, (100, 2))  # (S,N)
    tgt = randint(200, (150, 2))  # (T,N)
    mem = randn((100, 2, 256))  # (S,N,E)

    y_s2s = seq2seq(src, tgt)  # (T,N,C)
    y_enc = encoder(src)  # (S,N,C)
    y_dec = decoder(mem, tgt)  # (T,N,C)
    test = beam_search(y_enc, beam_probs := [1], src, 4)
    print(y_s2s.shape)
