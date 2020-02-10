#from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.enable_eager_execution()
import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from models import *
from utils import *
from dataloader import *

tf.keras.backend.clear_session()  # For easy reset of notebook state.
print(cv2.__version__)
print(tf.__version__)
assert tf.executing_eagerly() == True

#create DNN model

batch_size = 6
trainset_size = 6
testset_size = 6
epochs = 1000
dl = Dataloader(train_path = "./data/train", test_path= "./data/test", batch_size = batch_size)
train_generator, test_generator = dl.load_dl()
net = PatchRankingNet(dynamic = True)
net.compile(loss = None, optimizer =  keras.optimizers.Adam(learning_rate=lr), run_eagerly = True)
net.fit_generator(train_generator,
                                steps_per_epoch=trainset_size/batch_size,
                                epochs = epochs,
                                validation_data = test_generator,
                                validation_steps = testset_size/batch_size,
                                use_multiprocessing = True,
                                shuffle = True)

patches, additionals = dl.load_data(test=True)
pred = net(patches)
refs,snss,matches = additionals
imgs = register_with_predicted(pred, matches, refs, snss)
for i in range(imgs.shape[0]):
    cv2.imwrite("registered_cv2{}.jpg".format(i), imgs[i])
