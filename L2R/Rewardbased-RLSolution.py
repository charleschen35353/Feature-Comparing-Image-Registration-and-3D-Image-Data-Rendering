#!/usr/bin/env python
# coding: utf-8

# In[21]:


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow import keras

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2" #(or "1" or "2")

tf.keras.backend.clear_session()  # For easy reset of notebook state.
print(cv2.__version__)
print(tf.__version__)
#assert tf.executing_eagerly() == True


# In[3]:


IMG_WIDTH = 595
IMG_HEIGHT = 842
IMG_CHN = 3
BBOX_LENGTH = 21 

NUM_F_POINTS = 2000
NUM_MATCHES = 200
NUM_CHOSEN_MATCHES = 30

DUMMY_COOR_MIN = 0
DUMMY_COOR_MAX = 999

FD_DIM = 32
RANDOM_PROB = 0.5
F_DIST_MAX = 20
COOR_DIST_MAX = 100
IMG_CENTER_W = 298 
IMG_CENTER_Y = 421

LBL_BUILDING_EP = 100
VALID_DESCRIPTOR_COUNTS = 20

eps = 0.85
alpha = 5
beta = 0
lr = 1e-4
label_lr = 1e-2

assert NUM_F_POINTS > NUM_MATCHES
assert NUM_MATCHES > NUM_CHOSEN_MATCHES

def batch_match(d1s, d2s):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = []

    for i in range(len(d1s)):
        matches.append(matcher.match(d1s[i], d2s[i]))
    return matches


def calc_homographies(kp1s, kp2s, matches):
    # Define empty matrices of shape no_of_matches * 2. 
    homographies = []
    temp = matches
    matches = []
    for i in range(len(temp)):
        if len(temp[i])>0 :
            m = list(filter(None, temp[i]))
            matches.append(m)
        else:
            matches.append([])

    for i in range(len(matches)):
        matches[i].sort(key = lambda x: x.distance)
        matches[i] = matches[i][:int(len(matches[i])*1.0)]

        p1 = np.zeros((len(matches[i]), 2)) 
        p2 = np.zeros((len(matches[i]), 2)) 

        if len(matches[i]) > 7:
            for j in range(len(matches[i])):
                p1[j, :] = kp1s[i][matches[i][j].queryIdx].pt 
                p2[j, :] = kp2s[i][matches[i][j].trainIdx].pt 
            homography, _ = cv2.findHomography(p1, p2, cv2.RANSAC) 
        else:
            homography = np.array([[0,0,0],[0,0,0],[0,0,0]])*1.0

        homographies.append(homography)

    return homographies

def register_images(sns_imgs, homographies, img_size = (IMG_WIDTH,IMG_HEIGHT), save = False):
    # Use this matrix to transform the 
    # colored image wrt the reference image. 
    transformed_imgs = []
    for i in range(sns_imgs.shape[0]):
        transformed_img = cv2.warpPerspective(sns_imgs[i], 
                            homographies[i], img_size) 
        
        transformed_imgs.append(transformed_img)

    return transformed_imgs

def visualize_matches(ref_imgs, sns_imgs, kp1s, kp2s, matches):
    for i in range(ref_imgs.shape[0]):
        temp = matches
        matches = []
        for j in range(len(temp)):
            if temp[j] != []:
                matches.append(list(filter(None, temp[j])))

        imMatches = cv2.drawMatches(ref_imgs[i], kp1s[i], sns_imgs[i], kp2s[i], matches[i], None)
        cv2.imwrite("matches_{}.jpg".format(i), imMatches)
    
  
def extract_features(ref_image, sns_image):
    # Create ORB detector with 5000 features. 
    
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY) 
    sns_image = cv2.cvtColor(sns_image, cv2.COLOR_BGR2GRAY) 
    orb_detector = cv2.ORB_create(NUM_F_POINTS) 
    kp1, d1 = orb_detector.detectAndCompute(ref_image, None) 
    kp2, d2 = orb_detector.detectAndCompute(sns_image, None) 
    kp1_np, kp2_np = [], []
    for i in range(NUM_F_POINTS):
        if i < len(kp1):
            kp1_np.append(kp1[i])
        else:
            kp1_np.append(None)
            if d1 is not None:
                d1 = np.vstack((d1, np.zeros_like(d1[0:1]))) 
            else:
                d1 = np.zeros(shape = (1, FD_DIM))

        if i < len(kp2):
            kp2_np.append(kp2[i])
        else:
            kp2_np.append(None)
            if d2 is not None:
                d2 = np.vstack((d2, np.zeros_like(d2[0:1])))
            else:
                d2 = np.zeros(shape= (1, FD_DIM))     
   
    kp1_np, kp2_np = np.array(kp1_np) , np.array(kp2_np)
    d1 = d1.astype(np.uint8)
    d2 = d2.astype(np.uint8)
    return [kp1_np[:NUM_F_POINTS], kp2_np[:NUM_F_POINTS], d1[:NUM_F_POINTS], d2[:NUM_F_POINTS]]

def extract_feature_batch(refs, sns):
    output = []
    for i in range(refs.shape[0]):
        out = extract_features(refs[i], sns[i])
        for j in range(4):
            if len(output) < 4:
                output.append(np.expand_dims(out[j], axis=0))
            else: 
                output[j] = np.vstack( (output[j], np.expand_dims(out[j], axis=0)) )

    return output
    

def get_model_inputs(refs,sns):
    p_ref, p_sns, matches, kprs, kpss = get_match_info(refs, sns)
    p_ref = (p_ref.astype(np.float32) / 255.0 ).reshape((p_ref.shape[0]*p_ref.shape[1],  p_ref.shape[2], p_ref.shape[3], p_ref.shape[4]))
    p_sns = (p_sns.astype(np.float32) / 255.0).reshape((p_sns.shape[0]*p_sns.shape[1], p_sns.shape[2], p_sns.shape[3], p_sns.shape[4]))
    #matches = matches.reshape(matches.shape[0] * matches.shape[1])
    return [p_ref, p_sns, matches, kprs, kpss]

def patch_dist(p1,p2):

    return np.mean(((p1 - p2)**2)**0.5)

def get_central_coor(patch,img):
    
    W,H = patch.shape[0], patch.shape[1]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            print("{} {}".format(i,j))
            if i+W < img.shape[0] and j+H < img.shape[1] and patch_dist(img[i:i+W, j:j+H, :],patch) < 1:
                return (i+W/2, j + H/2)
            
    print("patch does not exist in img.")
    return None


def patch_dist(p1,p2):

    return np.mean(((p1 - p2)**2)**0.5)

def get_central_coor(patch,img):
    
    W,H = patch.shape[0], patch.shape[1]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            print("{} {}".format(i,j))
            if i+W < img.shape[0] and j+H < img.shape[1] and patch_dist(img[i:i+W, j:j+H, :],patch) < 1:
                return (i+W/2, j + H/2)
            
    print("patch does not exist in img.")
    return None

def visualize_corresponding_patches(p1, p2):
    for j in range(50):
        vis = (np.concatenate((p1[0][j], p2[0][j]), axis=1))*255
        cv2.imwrite("patch pair {}.jpg".format(j), vis)
        
def visualize_coords(img, c):
    for j in range(500):
        cv2.circle(img[0], (c[0][j][0], c[0][j][1]) , 1, (0, 0, 255), -1)
    cv2.imwrite("New feture img.jpg", img[0])

def extract_match_patches(ref_imgs, sns_imgs, kprs, kpss, drs, dss):
    '''
    output: N * NUM_MATCHES * 2 * PATCH_H * PATCH_W * CHN Example:(6, 500, 2, 6, 6, 3)
    '''
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    patches, matches, coors = [], [], []
    for i in range(ref_imgs.shape[0]): # batch size
        results = []
        if np.sum(dss[i]) > VALID_DESCRIPTOR_COUNTS:
            results = matcher.match(drs[i], dss[i])
        patch, match, coor = [], [], []
        for r in results:
            pair = []
            cp = []
            if len(patch) >= NUM_MATCHES: break

            if kprs[i][r.queryIdx] is not None:
                x, y = kprs[i][r.queryIdx].pt
                x, y = int(x), int(y) 
                dis = kprs[i][r.queryIdx].size
            else:
                x, y = DUMMY_COOR_MIN, DUMMY_COOR_MIN

            if  x-(BBOX_LENGTH-1) > 0 and y-(BBOX_LENGTH-1) > 0 \
                and x+(BBOX_LENGTH-1)/2 < IMG_WIDTH and y+(BBOX_LENGTH-1)/2 < IMG_HEIGHT:
                pair.append(ref_imgs[i][y-int((BBOX_LENGTH-1)/2):y+int((BBOX_LENGTH-1)/2)\
                                       ,x-int((BBOX_LENGTH-1)/2):x+int((BBOX_LENGTH-1)/2)])
                cp.append([x,y])
            if  kpss[i][r.trainIdx] is not None:
                x, y =  kpss[i][r.trainIdx].pt
                x, y = int(x), int(y)
                dis = kpss[i][r.trainIdx].size
            else:
                x, y = DUMMY_COOR_MAX, DUMMY_COOR_MAX

            if  x-(BBOX_LENGTH-1) > 0 and y-(BBOX_LENGTH-1) > 0 \
                and x+(BBOX_LENGTH-1)/2 < IMG_WIDTH and y+(BBOX_LENGTH-1)/2 < IMG_HEIGHT:
                pair.append(sns_imgs[i][y-int((BBOX_LENGTH-1)/2):y+int((BBOX_LENGTH-1)/2)\
                                       ,x-int((BBOX_LENGTH-1)/2):x+int((BBOX_LENGTH-1)/2)])
                cp.append([x,y])

            if len(pair) == 2:
                patch.append(np.array(pair))
                match.append(r)
                coor.append(np.array(cp))
                
        while len(patch) < NUM_MATCHES:
            patch.append(np.zeros((2,BBOX_LENGTH-1,BBOX_LENGTH-1, IMG_CHN)))
            match.append(None)
            coor.append(np.array([[DUMMY_COOR_MIN,DUMMY_COOR_MIN], [DUMMY_COOR_MAX,DUMMY_COOR_MAX]]))
           
        patch = np.array(patch)
        patches.append(patch)
        match = np.array(match)
        matches.append(match)
        coor = np.array(coor)
        coors.append(coor)

    patches = np.array(patches)
    matches = np.array(matches)   
    coors = np.array(coors)

    return [patches[:,:,0,:,:,:], patches[:,:,1,:,:,:], matches]#, coors[:,:,0,:], coors[:,:,0,:]]


def get_match_info(refs,sns):
    """
    returns in 255 scale
    """
    kprs, kpss, drs, dss = extract_feature_batch(refs,sns)
    p_ref, p_sns, matches =  extract_match_patches(refs, sns, kprs, kpss, drs, dss)
    return [p_ref, p_sns, kprs, kpss, matches]

def get_model_inputs(refs,sns):
    p_ref, p_sns, kprs, kpss , matches = get_match_info(refs, sns)
    p_ref = (p_ref.astype(np.float32) / 255.0 )
    p_sns = (p_sns.astype(np.float32) / 255.0 )
    #matches = matches.reshape(matches.shape[0] * matches.shape[1])
    return [p_ref, p_sns, kprs, kpss, matches]

def scramble(a, axis=-1):
    """
    Return an array with the values of `a` independently shuffled along the
    given axis
    """ 
    b = a.swapaxes(axis, -1)
    n = a.shape[axis]
    idx = np.random.choice(n, n, replace=False)
    b = b[..., idx]
    return b.swapaxes(axis, -1)

def label_fetching(inputs):
    patches, matches, kprs, kpss, refs, sns = inputs
    inds = np.argsort(np.ones(shape = (patches.shape[0], NUM_MATCHES)))
    inds = scramble(inds)
    mask = np.where(inds < NUM_CHOSEN_MATCHES, tf.ones_like(inds), tf.zeros_like(inds))
    selected_matches = np.where(mask > 0.5, matches, np.zeros_like(mask))
    selected_matches = np.reshape(selected_matches[selected_matches != 0], (-1, NUM_CHOSEN_MATCHES))
    homos = np.array(calc_homographies(kprs, kpss, selected_matches))
    aligned_imgs = register_images(sns, homos)

    y_true = []
    _,_, kp1s, kp2s, new_matches  = get_match_info(refs, aligned_imgs)
    coor_dists, feature_dists = [], []

    if new_matches is None:
        y_true = np.zeros_like(inds)
    else:
        for i in range(kp1s.shape[0]):
            coor_dis, feature_dis = [], []
            for r in new_matches[i]:
                if r is None:
                    continue
                #valid_count+=1
                if kp1s[i][r.queryIdx] is not None:
                    x1, y1 = kp1s[i][r.queryIdx].pt
                else:
                    x1, y1 = DUMMY_COOR_MIN, DUMMY_COOR_MIN
                if kp2s[i][r.trainIdx].pt is not None:
                    x2, y2 = kp2s[i][r.trainIdx].pt
                else:
                    x2, y2 = DUMMY_COOR_MAX, DUMMY_COOR_MAX

                
                dis = min(((x1-x2)**2 + (y1-y2)**2)**0.5,COOR_DIST_MAX)
                coor_dis.append(dis) #euc dist
                feature_dis.append(min(r.distance, F_DIST_MAX))

            avg_coor_dist = sum(coor_dis)/max(1,len(coor_dis))
            avg_f_dist = sum(feature_dis)/max(1,len(feature_dis))
            coor_dists.append(avg_coor_dist)
            feature_dists.append(avg_f_dist)

        for c_d, f_d in zip(coor_dists, feature_dists):
            y_true.append(np.ones_like(inds[0]) * (COOR_DIST_MAX/max(1,c_d)) )

        y_true = np.array(y_true) * mask

    return y_true

loss_obj = tf.keras.losses.MeanSquaredError()

def loss(model, inputs, labels, training):
    patches, matches, kprs, kpss, refs, sns = inputs
    y_pred = model(patches, training=training) # batch * matches indicating quality
    inds = tf.argsort(tf.argsort(y_pred, axis = -1 ,direction='DESCENDING'))
    if tf.random.uniform((1,)) < RANDOM_PROB :
        print("Random action selected.")
        inds = tf.random.shuffle(inds)

    #select top N
    mask = tf.where(inds < NUM_CHOSEN_MATCHES, tf.ones_like(inds), tf.zeros_like(inds)) #(6,500)
    mask = mask.numpy()
    selected_matches = np.where(mask > 0.5, matches, np.zeros_like(mask))
    selected_matches = np.reshape(selected_matches[selected_matches != 0], (-1, NUM_CHOSEN_MATCHES))
    homos = np.array(calc_homographies(kprs, kpss, selected_matches))
    aligned_imgs = register_images(sns, homos)
    
    label_temp = []
    _,_, kp1s, kp2s, new_matches  = get_match_info(refs, aligned_imgs)
    coor_dists, feature_dists = [], []

    if new_matches is None:
        label_temp = np.zeros_like(inds)
    else:
        for i in range(kp1s.shape[0]):
            coor_dis, feature_dis = [], []
            for r in new_matches[i]:
                if r is None:
                    continue
                #valid_count+=1
                if kp1s[i][r.queryIdx] is not None:
                    x1, y1 = kp1s[i][r.queryIdx].pt
                else:
                    x1, y1 = DUMMY_COOR_MIN, DUMMY_COOR_MIN
                if kp2s[i][r.trainIdx].pt is not None:
                    x2, y2 = kp2s[i][r.trainIdx].pt
                else:
                    x2, y2 = DUMMY_COOR_MAX, DUMMY_COOR_MAX

                
                dis = min(((x1-x2)**2 + (y1-y2)**2)**0.5,COOR_DIST_MAX)
                coor_dis.append(dis) #euc dist
                feature_dis.append(min(r.distance, F_DIST_MAX))

            avg_coor_dist = sum(coor_dis)/max(1,len(coor_dis))
            avg_f_dist = sum(feature_dis)/max(1,len(feature_dis))
            coor_dists.append(avg_coor_dist)
            feature_dists.append(avg_f_dist)

        for c_d, f_d in zip(coor_dists, feature_dists):
            label_temp.append(np.ones_like(inds[0]) * (COOR_DIST_MAX/max(1,c_d))  )

        label_temp = np.array(label_temp) * mask
        #coor_loss = mask * 
        #label_temep += coor_loss
        label_temp = tf.convert_to_tensor(label_temp)

    labels = labels + label_lr * (labels - label_temp)
    print("Avg predicted coordiate feature distance:{} (dist cap:{})".format(sum(coor_dists)/len(coor_dists), COOR_DIST_MAX))
    print(labels)
    print(y_pred)
    return loss_obj(y_true=labels, y_pred=y_pred)

def infer_with_pred(y_pred, inputs):
    patches, matches, kprs, kpss, refs, sns = inputs
    inds = tf.argsort(tf.argsort(y_pred, direction='DESCENDING'))   
    #select top 100
    mask = tf.where(inds < NUM_CHOSEN_MATCHES, tf.ones_like(inds), tf.zeros_like(inds)) #(6,500)
    mask = mask.numpy()
    
    selected_matches = np.where(mask > 0.5, matches, np.zeros_like(mask))
    selected_matches = np.reshape(selected_matches[selected_matches != 0], (-1, NUM_CHOSEN_MATCHES))
    homos = np.array(calc_homographies(kprs, kpss, selected_matches))
    aligned_imgs = register_images(sns, homos)
    for i in range(len(aligned_imgs)):
        cv2.imwrite('aligned_{}.jpg'.format(i), aligned_imgs[i]) 

def grad(model, inputs, labels):

    with tf.GradientTape() as  tape:
        loss_value = loss(model, inputs, labels, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


	

def generate_generator_multiple(generator, path, batch_size = 16, img_height = IMG_HEIGHT, img_width = IMG_WIDTH):

        gen_ref = generator.flow_from_directory(path,
                                              classes = ["ref"],
                                              target_size = (img_height,img_width),
                                              batch_size = batch_size,
                                              shuffle=False, 
                                              seed=7)

        gen_sns = generator.flow_from_directory(path,
                                              classes = ["sns"],
                                              target_size = (img_height,img_width),
                                              batch_size = batch_size,
                                              shuffle=False, 
                                              seed=7)
        while True:
                X1i = gen_ref.next() #in 255
                X2i = gen_sns.next()
                
                pr,ps, kprs, kpss, matches = get_model_inputs(X1i[0].astype(np.uint8), X2i[0].astype(np.uint8))
                #kprs, kpss, _, _ =  extract_feature_batch(X1i[0].astype(np.uint8), X2i[0].astype(np.uint8))
                patch_input = np.concatenate((pr,ps), axis = 4)
                patch_input = np.reshape(patch_input,(patch_input.shape[0], BBOX_LENGTH-1, BBOX_LENGTH-1, IMG_CHN*2*NUM_MATCHES))

                yield [patch_input, matches, kprs, kpss, X1i[0].astype(np.uint8), X2i[0].astype(np.uint8)], None #Yield both images and their mutual label


class Dataloader:
    def __init__(self, train_path, test_path, batch_size = 16):
        
        train_imgen = keras.preprocessing.image.ImageDataGenerator()
        test_imgen = keras.preprocessing.image.ImageDataGenerator()

        self.train_generator = generate_generator_multiple(generator=train_imgen,
                                               path = str(train_path),
                                               batch_size=batch_size)       

        self.test_generator = generate_generator_multiple(test_imgen,
                                              path = str(test_path),
                                              batch_size=batch_size)   
        #Label building
        labels = None
        for ep in range(LBL_BUILDING_EP):
             print("Label forming in progress... {} out of {}".format(ep+1, LBL_BUILDING_EP))
             x,_ =  next(self.train_generator) 
             y = label_fetching(x)
             if labels is None:
                 labels = y
             else:
                 labels += y
             
        self.labels = labels/ LBL_BUILDING_EP

    def load_data(self, test = False):
        if test:
            x, _ = next(self.test_generator)
            return x, self.labels
        else:
            x, _ = next(self.train_generator)
            return x, self.labels

    def load_dl(self):
        return [self.train_generator, self.test_generator]

batch_size = 6
trainset_size = 6
testset_size = 6
epochs = 20

dl = Dataloader(train_path = "./data/train", test_path= "./data/test", batch_size = batch_size)

model = keras.Sequential([
    #similarity value for each match
    layers.Input(shape = ( BBOX_LENGTH-1, BBOX_LENGTH-1, IMG_CHN*2*NUM_MATCHES), dtype = 'float32', name = "input_patches" ),
    
    #p = layers.Reshape((BBOX_LENGTH-1, BBOX_LENGTH-1, IMG_CHN*2*NUM_MATCHES))(patches_plh)
    layers.Conv2D(16, 3, strides=(1, 1), name = "pr_conv1",  padding='valid', activation="relu", kernel_initializer='glorot_uniform'),
    layers.MaxPool2D(),
    layers.Conv2D(64, 3, strides=(1, 1), name = "pr_conv2",  padding='valid', activation="relu", kernel_initializer='glorot_uniform'),
    layers.Conv2D(128, 3, strides=(1, 1), name = "pr_conv3", padding='valid', activation="relu", kernel_initializer='glorot_uniform'),
    layers.Conv2D(128, 3, strides=(1, 1), name = "pr_conv4", padding='valid', activation="relu", kernel_initializer='glorot_uniform'),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(NUM_MATCHES, activation= 'sigmoid', name = "pr_out", kernel_initializer = 'glorot_uniform')
])
 



print("Trainables:")
print(tf.trainable_variables)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
'''
inputs, labels = dl.load_data()

patch_input, matches, kprs, kpss, refs, sns = inputs
predict = model(patch_input)
l = loss(model, inputs, labels, training=False)
print("Loss test: {}".format(l))

loss_value, grads = grad(model, inputs, labels)

print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                          loss_value.numpy()))

optimizer.apply_gradients(zip(grads, model.trainable_variables))

print("Step: {}, Loss: {}".format(optimizer.iterations.numpy(),
                                          loss(model, inputs, labels, training=True).numpy()))
print(labels)
infer_with_pred(predict, inputs)
print("Infer images saved.")
'''

# Keep results for plotting
train_loss_results = []

num_epochs = 4000

for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()

    inputs, labels =  dl.load_data()
    pred = model(inputs[0])
    
    infer_with_pred(pred,inputs)
    
    loss_value, grads = grad(model, inputs, labels)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    epoch_loss_avg(loss_value)
    train_loss_results.append(epoch_loss_avg.result())
        
    print("Epoch {:03d}: Loss: {}".format(epoch, epoch_loss_avg.result()))
