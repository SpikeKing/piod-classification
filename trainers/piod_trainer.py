# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
import os
import warnings

import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV

from bases.trainer_base import TrainerBase
from models.piod_model import create_model
from utils.np_utils import prp_2_oh_array


class PiodTrainer(TrainerBase):

    def __init__(self, model, data, config):
        super(PiodTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.cp_dir,
                                      '%s.weights.{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp_name),
                monitor="val_loss",
                mode='min',
                save_best_only=True,
                save_weights_only=False,
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.tb_dir,
                write_images=True,
                write_graph=True,
            )
        )

        # self.callbacks.append(FPRMetric())
        self.callbacks.append(FPRMetricDetail())

    def train(self):
        X = self.data[0]
        Y = self.data[1]

        model_wrapper = KerasClassifier(
            build_fn=create_model,
            verbose=0
        )

        optimizers = ['rmsprop', 'adam']
        init_modes = ['normal', 'uniform']
        epochs = np.array([50, 100])
        batches = np.array([10, 20])

        param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init_mode=init_modes)
        grid = GridSearchCV(estimator=model_wrapper, param_grid=param_grid, n_jobs=4)
        grid_result = grid.fit(X, Y)

        print('[INFO] Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))

        for params, mean_score, scores in grid_result.grid_scores_:
            print('[INFO] %f (%f) with %r' % (scores.mean(), scores.std(), params))


class FPRMetric(Callback):
    """
    输出F, P, R
    """

    def on_epoch_end(self, batch, logs=None):
        val_x = self.validation_data[0]
        val_y = self.validation_data[1]

        prd_y = prp_2_oh_array(np.asarray(self.model.predict(val_x)))

        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            val_y, prd_y, average='macro')
        print " — val_f1: % 0.4f — val_pre: % 0.4f — val_rec % 0.4f" % (f_score, precision, recall)


class FPRMetricDetail(Callback):
    """
    输出F, P, R
    """

    def on_epoch_end(self, batch, logs=None):
        val_x = self.validation_data[0]
        val_y = self.validation_data[1]

        prd_y = prp_2_oh_array(np.asarray(self.model.predict(val_x)))

        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        precision, recall, f_score, support = precision_recall_fscore_support(val_y, prd_y)

        for p, r, f, s in zip(precision, recall, f_score, support):
            print " — val_f1: % 0.4f — val_pre: % 0.4f — val_rec % 0.4f - ins %s" % (f, p, r, s)
