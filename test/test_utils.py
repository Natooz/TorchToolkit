import torch

from torchtoolkit.utils import convert_idx_tensor, seed_everything


def test_convert_idx_tensor():
    # cases is a list of tuples of shape (tensor_to_index, idx_tensor, solution)
    cases = [(torch.Tensor([[0, 1, 2, 3], [4, 5, 6, 7]]),  # (2,4)
              torch.LongTensor([1, 2]),  # (2) --> [(0, 1), (1, 2)]
              torch.Tensor([1, 6])),  # (2)

             (torch.Tensor([[[0, 1, 2, 3], [4, 5, 6, 7]], [[8, 9, 10, 11], [12, 13, 14, 15]]]),  # (2,2,4)
              torch.LongTensor([[0, 1], [2, 3]]),  # (2,2) --> [[(0, 0, 1, 1)], [(0, 1, 0, 1)], [(0, 1, 2, 3)]]
              torch.Tensor([0, 5, 10, 15])),  # (4)

             (torch.Tensor([[[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
                             [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]],
                            [[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
                             [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]],
                            [[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
                             [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]]]),  # (3,4,2,5)
              torch.LongTensor([[[0, 2], [1, 3], [1, 2], [4, 3]],  # (3,4,2) --> [(0, 0, 0, 0, ...), (0, 0, 1, 1, ...),
                                [[0, 2], [1, 3], [1, 2], [4, 3]],  # (0, 1, 0, 1, ...),
                                [[0, 2], [1, 3], [1, 2], [4, 3]]]),  # (idx.flatten)]
              torch.Tensor([0, 7, 1, 3, 1, 2, 4, 3, 0, 2, 1, 3, 1, 2, 4, 3, 0, 2, 1, 3, 1, 2, 4, 3]))]  # (24)
    for tensor_to_index, idx_tensor, solution in cases:
        '''idx_conv = convert_idx_tensor(idx_tensor)  # uncomment for debug
        res = tensor_to_index[convert_idx_tensor(idx_tensor)]
        test = tensor_to_index[convert_idx_tensor(idx_tensor)] == solution'''
        assert torch.all(tensor_to_index[convert_idx_tensor(idx_tensor)] == solution)


def test_seed_everything():
    seed_everything(777)


if __name__ == '__main__':
    test_convert_idx_tensor()
