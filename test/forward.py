
from torch import randint, randn

from flexformer import Transformer


if __name__ == '__main__':
    seq2seq = Transformer(2, 2, 200, 256, 4, 512)
    encoder = Transformer(2, 0, 200, 256, 4, 512)
    decoder = Transformer(0, 2, 200, 256, 4, 512)

    src = randint(200, (100,))  # (S,N)
    tgt = randint(200, (150,))  # (T,N)
    mem = randn((100, 256))  # (S,N,E)

    y_s2s = seq2seq(src, tgt)  # (T,N,E)
    y_enc = encoder(src)  # (S,N,E)
    y_dec = decoder(mem, tgt)  # (T,N,E)
    print(y_s2s.shape)
