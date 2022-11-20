import unittest
from src.basic import *

import numpy as np

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)


    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)

        y.backward()
        expected = np.array(6.0) # 2*x
        self.assertEqual(x.grad, expected)


    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        numerical_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, numerical_grad)
        self.assertTrue(flg)

    