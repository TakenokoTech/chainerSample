# -*- coding: utf-8 -*-
from chainer import Chain
import chainer.links as L
import chainer.functions as F

class DLChain(Chain):

    def __init__(self):
        super(DLChain, self).__init__(
            l1 = L.Linear(1, 100),
            l2 = L.Linear(100, 30),
            l3 = L.Linear(30, 1)
        )

    def predict(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)
