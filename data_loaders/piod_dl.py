# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
import os

import numpy as np
from keras.utils import to_categorical

from bases.data_loader_base import DataLoaderBase
from root_dir import ROOT_DIR


class PiodDL(DataLoaderBase):
    def __init__(self, config=None):
        super(PiodDL, self).__init__(config)
        raw_path = os.path.join(ROOT_DIR, 'experiments', 'diabetes.csv')
        dataset = np.loadtxt(raw_path, delimiter=',', skiprows=1)
        print('[INFO] dataset: %s' % str(dataset.shape))

        self.X_train = dataset[:, :8]
        self.y_train = dataset[:, 8]

        self.X_test = np.array([])  # 暂时不用
        self.y_test = np.array([])

        print "[INFO] X_train.shape: %s, y_train.shape: %s" \
              % (str(self.X_train.shape), str(self.y_train.shape))
        print "[INFO] X_test.shape: %s, y_test.shape: %s" \
              % (str(self.X_test.shape), str(self.y_test.shape))

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test
