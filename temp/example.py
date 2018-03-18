# -*- coding: utf-8 -*-

# 数値計算関連
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from chainer import Chain, Variable, optimizers
import chainer.functions as F
import chainer.links as L
from DLChain import DLChain

# 乱数のシードを固定
random.seed(1)

# 標本データの生成
x, y = [], []
for i in np.linspace(-3, 3, 100):
    x.append([i])
    y.append([math.sin(i)])

# chainerの変数として再度宣言
x = Variable(np.array(x, dtype=np.float32))
y = Variable(np.array(y, dtype=np.float32))

# NNモデルを宣言
model = DLChain()

# 損失関数の計算
def forward(x, y, model):
    t = model.predict(x)
    loss = F.mean_squared_error(t, y)
    return loss

# chainerのoptimizer
optimizer = optimizers.Adam()

# modelのパラメータをoptimizerに渡す
optimizer.setup(model)

# パラメータの学習を繰り返す
for i in range(0, 1000):
    loss = forward(x, y, model)
    print("{0:.20f}".format(loss.data)) 
    optimizer.update(forward, x, y, model)

# プロット
t = model.predict(x)
# 標本
plt.plot(x.data, y.data, c='green')
# DLの結果
plt.scatter(x.data, t.data, c='red')
plt.grid(which='major', color='gray', linestyle='-')
plt.ylim(-1.5, 1.5)
plt.xlim(-4, 4)
plt.show()
