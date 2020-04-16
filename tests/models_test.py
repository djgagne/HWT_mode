from hwtmode.models import BaseConvNet
import unittest
import numpy as np

class TestModels(unittest.TestCase):
    def setUp(self) -> None:
        self.x_shape = [32, 32, 3]
        self.train_size = 128
        self.x = np.random.normal(size=[self.train_size] + self.x_shape).astype(np.float32)
        self.y = np.random.randint(0, 2, size=self.train_size)

    def test_network_build(self):
        bcn = BaseConvNet(min_filters=4, filter_growth_rate=1.5, min_data_width=8,
                          dense_neurons=8, output_type="sigmoid")
        bcn.build_network(self.x_shape, 1)
        assert bcn.model_.layers[1].output.shape[-1] == bcn.min_filters
        assert bcn.model_.layers[-6].output.shape[1] == bcn.min_data_width
        return

    def test_saliency(self):
        bcn = BaseConvNet(min_filters=4, filter_growth_rate=1.5, min_data_width=8,
                          dense_neurons=8, output_type="sigmoid")
        bcn.build_network(self.x_shape, 1)
        sal = bcn.saliency(self.x)
        assert sal.max() > 0
        self.assertListEqual(list(sal.shape[1:]), list(self.x.shape))