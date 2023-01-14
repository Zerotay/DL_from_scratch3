import numpy as np

# 벡터의 내적
a = np.array([1,2,3])
b = np.array([4,5,6])
c = np.dot(a,b)
print(c)

# 행렬의 곱
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
c = np.dot(a,b)
print(c)

from dezero import Variable
import dezero.functions as F 

x = Variable(np.random.randn(2,3))
W = Variable(np.random.randn(3,4))
y = F.matmul(x, W)
y.backward()

print(x.grad.shape)
print(W.grad.shape)