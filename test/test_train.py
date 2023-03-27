

def test_train():
    import logging

    from torch import randint, LongTensor
    from torch.nn import TransformerEncoder, TransformerEncoderLayer, Embedding, Linear, Module
    from torch.utils.data import Dataset

    from torchtoolkit.train import log_cuda_info, log_model_parameters
    from torchtoolkit.data import create_subsets
    from torchtoolkit.sampling import top_k

    nb_classes = 200

    class MyDataset(Dataset):  # random dataset for test
        def __init__(self, nb_samples: int):
            self.nb_samples = nb_samples

        def __getitem__(self, index):
            return randint(0, nb_classes, (30, )), randint(0, nb_classes, (30, ))

        def __len__(self):
            return self.nb_samples

    class MyModel(Module):
        def __init__(self):
            super().__init__()
            self.embedding = Embedding(nb_classes, 64)
            self.encoder = TransformerEncoder(TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128,
                                                                      batch_first=True), 2)
            self.to_logits = Linear(64, nb_classes, bias=False)

        def forward(self, x: LongTensor):
            return self.to_logits(self.encoder(self.embedding(x)))

        def forward_train(self, x: LongTensor, target: LongTensor, crit: Module, k: int = 5):
            logits = self.forward(x)  # (N,T,C)
            loss = crit(logits.transpose(2, 1), target)  # (N,C,T) and (N,T)
            y_sampled = top_k(logits.transpose(1, 0), k)  # (N,T)
            return logits, loss, y_sampled

    # logging.basicConfig(format="")
    logger = logging.getLogger('test_train')
    (fh := logging.FileHandler(filename="training.log")).setLevel(logging.DEBUG)
    (sh := logging.StreamHandler()).setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG)

    dataset = MyDataset(500)
    _, _ = create_subsets(dataset, [0.3])

    model = MyModel()
    log_cuda_info(logger=logger)
    log_model_parameters(model, logger=logger)


if __name__ == '__main__':
    test_train()
