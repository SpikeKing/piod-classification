# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
import os
from keras import Input, Model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model

from bases.model_base import ModelBase


class PiodModel(ModelBase):
    """
    SimpleMnist模型
    """

    def __init__(self, config=None):
        super(PiodModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        model = create_model()
        if self.config:
            plot_model(model, to_file=os.path.join(self.config.img_dir, "model.png"), show_shapes=True)  # 绘制模型图
        self.model = model


def create_model(optimizer='adam', init_mode='uniform'):
    main_input = Input(shape=(8,), name='main_input')
    output = Dense(units=12, kernel_initializer=init_mode, activation='relu')(main_input)
    output = Dense(units=8, kernel_initializer=init_mode, activation='relu')(output)
    output = Dense(units=1, kernel_initializer=init_mode, activation='sigmoid')(output)
    model = Model([main_input], output)

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model
