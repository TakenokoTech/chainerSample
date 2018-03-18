import os
import json
import pickle
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain
from chainer.links.caffe import CaffeFunction
from utils.log import log

class Illust2Vec(Chain):

    CAFFEMODEL_FN = './dataset/Illustration2Vec/illust2vec_ver200.caffemodel'
    # CAFFEMODEL_FN = './dataset/bvlc_googlenet.caffemodel'
    PKL_FN = './build/Illustration2Vec/illust2vec_ver200.pkl'

    def __init__(self, n_classes, unchain=True):
        log.info("Illust2Vec.__init__ called.")
        w = chainer.initializers.HeNormal()
        if 1: # not os.path.exists(self.PKL_FN):
            with open(self.CAFFEMODEL_FN, 'rb') as model_file: log.debug(f'tuning start. {len(model_file.read())}')
            model = CaffeFunction(self.CAFFEMODEL_FN)
            # binary = pickle.dumps(model)
            # with open(self.PKL_FN, 'wb') as pkl_file:
            #     pickle.dump(model, pkl_file)
        else:
            log.debug("tuning not start.")
            # model = pickle.load(open(self.PKL_FN, 'rb'))
        log.debug("tuning success.")
        del model.encode1
        del model.encode2
        del model.forwards['encode1']
        del model.forwards['encode2']
        model.layers = model.layers[:-2]

        super(Illust2Vec, self).__init__()
        with self.init_scope():
            self.trunk = model
            self.fc7 = L.Linear(None, 4096, initialW=w)
            self.bn7 = L.BatchNormalization(4096)
            self.fc8 = L.Linear(4096, n_classes, initialW=w)
        log.info(f'Illust2Vec.__init__ finished.')

    def __call__(self, x):
        log.info("Illust2Vec.__call__ called.")
        h = self.trunk({'data': x}, ['conv6_3'])[0] 
        h.unchain_backward()
        h = F.dropout(F.relu(self.bn7(self.fc7(h))))
        log.info(f'Illust2Vec.__call__ finished. {len(self.fc8(h))}')
        return self.fc8(h)

