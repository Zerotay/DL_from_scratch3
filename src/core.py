import numpy as np
from typing import List, Tuple
# class Function: pass



class Variable:
    def __init__(self, data: np.ndarray):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'Only support {np.ndarray}')

        self.data = data
        self.grad = None
        self.creator = None # prev pointer
        self.generation

    def set_creator(self, func: 'Function'): 
        # https://stackoverflow.com/questions/55320236/does-python-evaluate-type-hinting-of-a-forward-reference
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data) # 편미분의 형상 같아야

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            for x, gx in zip(f.inputs, gxs):
                x.grad = gx

                if x.creator is not None:
                    funcs.append(x.creator)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs: Tuple[Variable]) -> Variable:
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys, )

        outputs = [Variable(as_array(y)) for y in ys] # as_array 를 통해 x ** 2 등이 scalar 되는 것 방지
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else output[0]

    def forward(self, xs: Tuple[np.ndarray]) -> np.ndarray:
        raise NotImplementedError()

    def backward(self, gys: Tuple[np.ndarray]) -> np.ndarray:
        raise NotImplementedError()



class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x ** 2

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


class Add(Function):
    def forward(self, xs: List[np.ndarray]) -> np.ndarray:
        x0, x1 = xs
        y = x0 + x1
        return y


def numerical_diff(f: Function, x: Variable, eps: float=1e-4) -> np.ndarray:
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data-y0.data) / (2*eps)


def square(x: Variable) -> Variable:
    f = Square()
    return f(x)


def exp(x: Variable) -> Variable:
    f = Exp()
    return f(x)


def add(x0: Variable, x1: Variable) -> Variable:
    f = Add()
    return f(x0, x1)