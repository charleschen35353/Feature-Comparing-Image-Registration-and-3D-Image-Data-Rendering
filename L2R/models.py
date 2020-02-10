import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from utils import *

DIST_THRES = 100
RANDOM_PROB = 0.15
eps = 0.9
lr = 5e-4


def register_with_predicted(pred, matches, refs, snss):
    inds = np.argsort(pred, axis = 1)
    
    if np.random.uniform() < RANDOM_PROB * eps:#**(self.iteration / 100):
        inds = np.random.shuffle(inds)
    
    mask = np.where(inds < 50, np.ones_like(inds), np.zeros_like(inds)) # -1*NUM_MATCHES
    selected_matches = np.where( mask >= 0.5, matches, np.zeros_like(matches)  )
    good_matches = []
    
    for i in range(selected_matches.shape[0]):
        good_matches.append(selected_matches[i][selected_matches[i] != 0])

    kprs, kpss, _, _ =  extract_feature_batch(refs,snss)
    homo = calc_homographies(kprs, kpss, good_matches)
    imgs = np.array(register_images(snss, homo))
    
    return imgs, mask

def registration_loss(y_pred, imgs):

    
    ##############numpy operation###############
    refs, snss = imgs[:,:,:,:3], imgs[:,:,:,3:]
    _, _, matches, _, _ = get_model_inputs(refs, snss)
    

    pred = y_pred.numpy()
    imgs, mask = register_with_predicted(pred, matches, refs, snss)

    _, _, new_matches, kp1s, kp2s = get_match_info(refs, imgs)

    feature_diss,coor_diss, gt = [], [], []
    
    for i in range(kp1s.shape[0]):
        coor_dis, feature_dis, valid_count = 0, 0, 0
        for r in new_matches[i]:
            if r is None:
                continue
            valid_count+=1
            x1, y1, _ = kp1s[i][r.queryIdx]
            x2, y2, _ = kp1s[i][r.trainIdx]
            coor_dis += ((x1-x2)**2 + (y1-y2)**2)**0.5 #euc dist
            feature_dis += r.distance
            
        coor_diss.append(coor_dis/valid_count)
        feature_diss.append(feature_dis/valid_count)

    coor_diss = np.array(coor_diss)
    feature_diss = np.array(feature_diss)

    for c_d, f_d in zip(coor_diss, feature_diss):
        gt.append(np.ones_like(pred[0]) * 1/(c_d+f_d) )
    gt = np.array(gt)
    #################### END ####################
    #ground truth here: gt
    gt = tf.convert_to_tensor(gt)
    loss = tf.reduce_mean(((gt - y_pred)*mask)**2)
    
    return loss
    #def MSE(y_pred, y_true):
    #    loss = tf.reduce_mean(((gt - y_pred)*mask)**2)
    #    return loss
    
    #return MSE

class PatchRankingNet(tf.keras.Model):
    
    def __init__(self, dynamic=True):
        super(PatchRankingNet, self).__init__(dynamic=dynamic)
        self.conv1 = layers.Conv2D(16, 3 ,strides=(2, 2), name = "conv1",\
                                padding='valid', activation="relu", kernel_initializer='glorot_uniform')
        self.conv2 = layers.Conv2D(64, 3, strides=(1, 1), name = "conv2",\
                                padding='valid', activation="relu", kernel_initializer='glorot_uniform')
        self.conv3 = layers.Conv2D(128, 3, strides=(1, 1), name = "conv3",\
                                padding='valid', activation="relu", kernel_initializer='glorot_uniform')
        self.flat = layers.Flatten()
        self.fc1 = layers.Dense(32, activation= 'relu', name = "fc1", kernel_initializer = 'glorot_uniform')
        self.fc2 = layers.Dense(1, activation= 'sigmoid', name = "fc2", kernel_initializer = 'glorot_uniform')
        
    def call(self, inputs, training=True, mask=None):
        x, imgs = inputs
        imgs = imgs.numpy()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = tf.reshape(x, (-1, NUM_MATCHES))
        self.add_loss(registration_loss(x,imgs))
        return x


#create DNN model







