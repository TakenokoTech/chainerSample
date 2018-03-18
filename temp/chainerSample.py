import numpy as np
import chainer
import chainer.links as L

x_data = np.array(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
x = chainer.Variable(x_data)
y = x ** 2 - 2 * x + 1

# 勾配計算の前には初期化
y.grad = np.ones((2, 3), dtype=np.float32)
y.backward()
# print(x.grad)

# ランダムに初期値
f = L.Linear(3, 2)
x = chainer.Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
y = f(x)
# print(f.W.data)
# print(f.b.data)

f.zerograds()

x = chainer.Variable(np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32))
l1 = L.Linear(4, 3)
l2 = L.Linear(3, 2)
h = l1(x)
y = l2(h)
# print(y.data)

