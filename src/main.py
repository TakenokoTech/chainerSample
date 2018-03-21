import os
import glob
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from chainer.datasets import LabeledImageDataset
from chainer.datasets import TransformDataset
from chainer.dataset import concat_examples
from chainer.training import extensions
from chainer.training import triggers
from chainer import datasets
from chainer import iterators
from chainer import training
from chainer import optimizers
from chainer import optimizer
from chainer import serializers
from chainer import config
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from IPython.display import Image

from model.train import calcMean
from model.train import transform
from model.train import lr_drop

from common.log import log
from model.Illust2Vec import Illust2Vec
from const import CONST

if __name__ == '__main__':

    # ログファイルリセット
    log.clear()
    log.info("__name__ called.")

    # buildフォルダ作成
    os.makedirs("log", exist_ok=True)
    os.makedirs("build/result", exist_ok=True)
    os.makedirs("build/imageMean", exist_ok=True)
    os.makedirs("build/Illustration2Vec", exist_ok=True)

    # 各キャラクター
    dnames = glob.glob('{}/*'.format(CONST.IMG_DIR))

    # 画像ファイルパス一覧
    fnames = [glob.glob('{}/*.png'.format(d)) for d in dnames if not os.path.exists('{}/ignore'.format(d))]
    fnames = list(chain.from_iterable(fnames))
    # log.debug(f'fnames = {fnames}')

    # それぞれにフォルダ名から一意なIDを付与
    labels = [os.path.basename(os.path.dirname(fn)) for fn in fnames]
    dnames = [os.path.basename(d) for d in dnames if not os.path.exists('{}/ignore'.format(d))]
    labels = [dnames.index(l) for l in labels]

    # データセット作成
    dataset = LabeledImageDataset(list(zip(fnames, labels)))
    calcMean(dataset)

    # 変換付きデータセットにする
    transData = TransformDataset(dataset, transform)

    # データセットを学習用と検証用に分ける
    train, valid = datasets.split_dataset_random(transData, int(len(dataset) * 0.8), seed=0)

    # モデルインポート
    n_classes = len(dnames)
    model = Illust2Vec(n_classes)
    model = L.Classifier(model)

    # 訓練セットアップ
    train_iter = iterators.MultiprocessIterator(train, CONST.BATCH_SIZE)
    valid_iter = iterators.MultiprocessIterator(valid, CONST.BATCH_SIZE, repeat=False, shuffle=False)
    opt = optimizers.MomentumSGD(lr=CONST.INITIAL_LR)
    opt.setup(model)
    opt.add_hook(optimizer.WeightDecay(0.0001))
    updater = training.StandardUpdater(train_iter, opt, device=CONST.GPU_ID)

    # 訓練パラメータ
    trainer = training.Trainer(updater, (CONST.TRAIN_EPOCH, 'epoch'), out='build/result')
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time', 'lr']))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'))
    trainer.extend(extensions.Evaluator(valid_iter, model, device=CONST.GPU_ID))
    trainer.extend(lr_drop, trigger=triggers.ManualScheduleTrigger(CONST.LR_DROP_EPOCH, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.observe_lr())
    trainer.run()
    
    Image(filename='build/result/loss.png')
    Image(filename='build/result/accuracy.png')

    # 前の精度と比べて更新
    pastAccuracy = 0
    currentAccuracy = 0
    if os.path.exists(CONST.RESULT_MODEL) and os.path.exists(CONST.RESULT_LOG) :
        with open(CONST.RESULT_LOG , 'r') as f:
            j = json.load(f) 
            pastAccuracy = j[-1]['main/accuracy']
    if os.path.exists(CONST.NEW_LOG) :
        with open(CONST.NEW_LOG , 'r') as f:
            j = json.load(f) 
            currentAccuracy = j[-1]['main/accuracy']
    log.info('pastAccuracy: {0}, currentAccuracy: {1}'.format(pastAccuracy, currentAccuracy))
    if pastAccuracy > currentAccuracy:
        serializers.load_npz(CONST.RESULT_MODEL, model) 
    else:
        shutil.copy(CONST.NEW_LOG, CONST.RESULT_LOG)
        serializers.save_npz(CONST.RESULT_MODEL, model) 

    # 結果
    config.train = False
    correctAnswerCount = 0
    for _ in range(COUNT.CHECK_COUNT):
        x, t = valid[np.random.randint(len(valid))]
        x = cuda.to_gpu(x)
        y = F.softmax(model.predictor(x[None, ...]))
        pred = os.path.basename(dnames[int(y.data.argmax())])
        label = os.path.basename(dnames[t])
        log.info('[{0:<2}] {1:<6}, pred: {2:<30} , label: {3:<30}'.format(_+1, (pred == label), pred, label))
        # x = cuda.to_cpu(x)
        # x += mean[:, None, None]
        # x = x / 256
        # x = np.clip(x, 0, 1)
        # plt.imshow(x.transpose(1, 2, 0))
        # plt.show()
        if (pred == label): correctAnswerCount += 1
    
    # 結果
    log.info('RESULT {0} / {1} ({2:.2%})'.format(correctAnswerCount, COUNT.CHECK_COUNT, (correctAnswerCount/COUNT.CHECK_COUNT)))