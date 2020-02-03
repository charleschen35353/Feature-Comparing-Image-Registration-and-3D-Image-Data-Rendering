from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
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

lr = 1e-3
batch_size = 6
trainset_size = 6
testset_size = 6
epochs = 10000

patch_ranker = PatchRanker(dynamic=True)
optimizer = keras.optimizers.Adam(learning_rate=lr)
patch_ranker.compile(optimizer = optimizer,run_eagerly = True)

dl = Dataloader(train_path = "./data/train", test_path= "./data/test", batch_size = batch_size)
train_generator, test_generator = dl.load_dl()
checkpoint_path = "L2R/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
#print(patch_ranker)

patch_ranker.fit_generator(train_generator,
                                steps_per_epoch=trainset_size/batch_size,
                                epochs = epochs,
                                validation_data = test_generator,
                                validation_steps = testset_size/batch_size,
                                use_multiprocessing = True,
                                shuffle=False,
				callbacks = [cp_callback])

