from pathlib import Path

import torch

from torchtoolkit.metrics import Accuracy


def test_accuracy_metric():
    def assert_equal(metr: Accuracy, file_name: str):
        res_save = metr.results
        metr.save(Path('test', file_name))
        metr.load(Path('test', file_name))
        assert res_save == metric.results, 'Accuracy metric failed, error in save / load'

    modes = [('greedy', {}), ('top_k', {'top_kp': 5}), ('top_p', {'top_kp': 0.9}), ('softmax', {}), ('likelihood', {})]
    for mode, params in modes:
        metric = Accuracy(mode=mode, **params)
        for _ in range(10):
            res = torch.randn((16, 32))
            expected = torch.randn((16, 32)) if mode == 'likelihood' else torch.randint(0, 32, (16, ))
            metric(res, expected)  # saving results
        assert_equal(metric, f'acc_{mode}.csv')

    metric = Accuracy('none')
    metric.results = torch.randn((10, 10)).tolist()
    assert_equal(metric, 'acc_none.csv')


if __name__ == '__main__':
    test_accuracy_metric()
