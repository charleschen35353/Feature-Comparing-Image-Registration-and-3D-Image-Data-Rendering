#!/usr/bin/env python
# coding: utf-8

# In[21]:


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow import keras
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2" #(or "1" or "2")

tf.keras.backend.clear_session()  # For easy reset of notebook state.
print(cv2.__version__)
print(tf.__version__)
#assert tf.executing_eagerly() == True


# In[3]:


IMG_WIDTH = 620
IMG_HEIGHT = 877
IMG_CHN = 3
NUM_F_POINTS = 5000
NUM_MATCHES = 200
BBOX_LENGTH = 21    


# In[4]:


import numpy as np
import cv2

NUM_F_POINTS = 2000

# Batch processing of cv2 non-learning based method
def extract_features(ref_image, sns_image):
    # Create ORB detector with 5000 features. 

    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY) 
    sns_image = cv2.cvtColor(sns_image, cv2.COLOR_BGR2GRAY) 
    orb_detector = cv2.ORB_create(5000) 
    kp1, d1 = orb_detector.detectAndCompute(ref_image, None) 
    kp2, d2 = orb_detector.detectAndCompute(sns_image, None) 
    return [kp1,kp2,d1,d2]

def extract_feature_batch(refs, sns):
    output = [[],[],[],[]]
    for i in range(refs.shape[0]):
        out = extract_features(refs[i], sns[i])
        for j in range(4):
            output[j].append(out[j])
    
    return output

def batch_match(d1s, d2s):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = []

    for i in range(len(d1s)):
        matches.append(matcher.match(d1s[i], d2s[i]))
    return matches

def validate_match(matches):
    for i in range(len(matches)):
        matches[i].sort(key = lambda x: x.distance)
        matches[i] = matches[i][:int(len(matches)*60)]
    return matches

def calc_homographies(kp1s, kp2s, matches):
    # Define empty matrices of shape no_of_matches * 2. 
    homographies = []
    temp = matches
    matches = []

    for i in range(len(temp)):
        if temp[i] != []:
            matches.append(list(filter(None, temp[i])))
            
    matches = validate_match(matches)
    
    for i in range(len(matches)):
        p1 = np.zeros((len(matches[i]), 2)) 
        p2 = np.zeros((len(matches[i]), 2)) 
        if p1.shape[0] != 0:
            for j in range(len(matches[i])):
                p1[j, :] = kp1s[i][matches[i][j].queryIdx].pt 
                p2[j, :] = kp2s[i][matches[i][j].trainIdx].pt 
            homography, _ = cv2.findHomography(p1, p2, cv2.RANSAC) 
        else:
            homography = np.zeros([3,3])
        homographies.append(homography)
    return homographies

def register_images(sns_imgs, homographies, img_size = (IMG_WIDTH,IMG_HEIGHT), save = False):
    # Use this matrix to transform the 
    # colored image wrt the reference image. 
    transformed_imgs = []
    for i in range(sns_imgs.shape[0]):
        transformed_img = cv2.warpPerspective(sns_imgs[i], 
                            homographies[i], img_size) 
        if save: 
            cv2.imwrite('aligned_{}.jpg'.format(i), transformed_img) 
        
        transformed_imgs.append(transformed_img)
    return transformed_imgs

def visualize_matches(ref_imgs, sns_imgs, kp1s, kp2s, matches):
    for i in range(ref_imgs.shape[0]):
        imMatches = cv2.drawMatches(ref_imgs[i], kp1s[i], sns_imgs[i], kp2s[i], matches[i], None)
        cv2.imwrite("matches_{}.jpg".format(i), imMatches)
    
    
def get_alignment_matrix(kprs, kpss, drs, dss):
    '''
    Inputs
        kprs: F keypoints for each reference(query) image of shape N*F*3 with X,Y,Size
        kpss: F keypoints for each sensed(train) image of shape N*F*3 with X,Y,Size
        drs: F feature descriptors for each reference(query) image of shape N*F*32 
        dss: F feature descriptors for each sensed(train) image of shape N*F*32 
    Output
        aligned feature points and correlated distance of size N*f*7 X1,Y1,Size1, X2, Y2, Size2, Distance
    '''

    alignment_matrices = None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    for i in range(kprs.shape[0]):
        results = matcher.match(d1s[i], d2s[i])
        aligned = []
        for r in results:
            if len(aligned) >= NUM_MATCHES: break
            temp = np.concatenate((kprs[i][r.queryIdx], kpss[i][r.trainIdx]))
            aligned.append(np.concatenate((temp, [r.distance])))
            
        while len(aligned) < NUM_MATCHES:
            aligned.append([0,0,0,0,0,0,0])
            
        aligned = np.array([aligned])
        if alignment_matrices is None:
            alignment_matrices = aligned
        else:
            alignment_matrices = np.vstack((alignment_matrices, aligned))
            
    #sample alignment_matrix
    alignment_matrices = np.rint(alignment_matrices).astype(np.uint32)
    return alignment_matrices
    
    
def extract_feature_coordinates(ref_image, sns_image):
    # Create ORB detector with 5000 features. 

    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY) 
    sns_image = cv2.cvtColor(sns_image, cv2.COLOR_BGR2GRAY) 
    orb_detector = cv2.ORB_create(NUM_F_POINTS) 
    kp1, d1 = orb_detector.detectAndCompute(ref_image, None) 
    kp2, d2 = orb_detector.detectAndCompute(sns_image, None) 
    kp1_np, kp2_np = [], []
    for i in range(NUM_F_POINTS):
        if i < len(kp1):
            kp1_np.append([kp1[i].pt[0],kp1[i].pt[1], kp1[i].size ] )
        else:
            kp1_np.append([0,0,0])
            d1 = np.vstack((d1, [np.zeros(32, dtype = np.uint8 )]))
            
        if i < len(kp2):
            kp2_np.append([kp2[i].pt[0],kp2[i].pt[1], kp2[i].size ] )
        else:
            kp2_np.append([0,0,0])
            d2 = np.vstack((d2, [np.zeros(32,dtype = np.uint8 )]))
        
    kp1_np, kp2_np = np.array(kp1_np) , np.array(kp2_np)

    return [kp1_np, kp2_np, d1, d2]

def extract_feature_coor_batch(refs, sns):
    output = []
    for i in range(refs.shape[0]):
        out = extract_feature_coordinates(refs[i], sns[i])
        for j in range(4):
            if len(output) < 4:
                output.append(np.expand_dims(out[j], axis=0))
            else: 
                output[j] = np.vstack( (output[j],np.expand_dims(out[j], axis=0)) )

    return output
    

    
def get_model_inputs(refs,sns):
    p_ref, p_sns, matches, kprs, kpss = get_match_info(refs, sns)
    p_ref = (p_ref.astype(np.float32) / 255.0 ).reshape((p_ref.shape[0]*p_ref.shape[1],                                                        p_ref.shape[2], p_ref.shape[3], p_ref.shape[4]))
    p_sns = (p_sns.astype(np.float32) / 255.0).reshape((p_sns.shape[0]*p_sns.shape[1],                                                        p_sns.shape[2], p_sns.shape[3], p_sns.shape[4]))
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


# In[104]:


def extract_match_patches(ref_imgs, sns_imgs, kprs, kpss, drs, dss):
    '''
    output: N * NUM_MATCHES * 2 * PATCH_H * PATCH_W * CHN Example:(6, 500, 2, 6, 6, 3)
    '''
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    patches, locations = [], []
    mask = np.zeros_like(ref_imgs)
    for i in range(kprs.shape[0]):
        results = matcher.match(drs[i], dss[i])
        patch = []
        coors = []
        for r in results:
            pair = []
            coor = []
            if len(patch) >= NUM_MATCHES: break
            x, y, dis = kprs[i][r.queryIdx].astype(np.uint32)
            if  x-(BBOX_LENGTH-1) > 0 and y-(BBOX_LENGTH-1) > 0                 and x+(BBOX_LENGTH-1)/2 < IMG_WIDTH and y+(BBOX_LENGTH-1)/2 < IMG_HEIGHT:
                pair.append(ref_imgs[i][y-int((BBOX_LENGTH-1)/2):y+int((BBOX_LENGTH-1)/2)                                       ,x-int((BBOX_LENGTH-1)/2):x+int((BBOX_LENGTH-1)/2)])
                coor.append([y,x])
                
            x, y, dis = kpss[i][r.trainIdx].astype(np.uint32)
            if  x-(BBOX_LENGTH-1) > 0 and y-(BBOX_LENGTH-1) > 0                 and x+(BBOX_LENGTH-1)/2 < IMG_WIDTH and y+(BBOX_LENGTH-1)/2 < IMG_HEIGHT:
                pair.append(sns_imgs[i][y-int((BBOX_LENGTH-1)/2):y+int((BBOX_LENGTH-1)/2)                                       ,x-int((BBOX_LENGTH-1)/2):x+int((BBOX_LENGTH-1)/2)])
                coor.append([y,x])
                
            if len(pair) == 2:
                patch.append(np.array(pair))
                coors.append(np.array(coor))
                mask[i][y-int((BBOX_LENGTH-1)/2):y+int((BBOX_LENGTH-1)/2)                                       ,x-int((BBOX_LENGTH-1)/2):x+int((BBOX_LENGTH-1)/2)] = 1

        while len(patch) < NUM_MATCHES:
            patch.append(np.zeros((2,BBOX_LENGTH-1,BBOX_LENGTH-1, IMG_CHN)))
            coors.append(np.array([[0,0],[0,0]]))
        patch = np.array(patch)
        patches.append(patch)
        coors = np.array(coors)
        locations.append(coors)

    patches = np.array(patches)
    locations = np.array(locations)
    return [patches[:,:,0,:,:,:], patches[:,:,1,:,:,:], locations[:,:,0,:], locations[:,:,1,:], mask ]


def get_match_info(refs,sns):
    """
    returns in 255 scale
    """
    kprs,kpss, drs, dss = extract_feature_coor_batch(refs,sns)
    p_ref, p_sns, coor_ref, coor_sns, mask =  extract_match_patches(refs, sns, kprs, kpss, drs, dss)
    return [p_ref, p_sns, coor_ref, coor_sns, mask]
    
def get_model_inputs(refs,sns):

    p_ref, p_sns, coor_ref, coor_sns, mask = get_match_info(refs, sns)
    p_ref = (p_ref.astype(np.float32) / 255.0 )
    p_sns = (p_sns.astype(np.float32) / 255.0 )
    coor_ref = (coor_ref.astype(np.float32) )
    coor_sns = (coor_sns.astype(np.float32)  )
    #matches = matches.reshape(matches.shape[0] * matches.shape[1])
    return [p_ref, p_sns, coor_ref, coor_sns, mask]


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
                X1i = gen_ref.next()#in 255
                X2i = gen_sns.next()
                
                pr,ps,cpr,cps,mask = get_model_inputs(X1i[0].astype(np.uint8), X2i[0].astype(np.uint8))
                # x,y in shape of batch_size * NUM_MATCHES* BOX_H * BOX_W * CHN
                # cx,cy in shape of batch_size * NUM_MATCHES * 2(x,y)
                patch_input = np.concatenate((pr,ps), axis = 4)
                patch_input = np.reshape(patch_input,(patch_input.shape[0], BBOX_LENGTH-1, BBOX_LENGTH-1, IMG_CHN*2*NUM_MATCHES))
                #imgs = np.concatenate((X1i[0], X2i[0]), axis = 3)
                coors = np.concatenate((cpr,cps), axis = -1)
                cv2.imwrite("mask.jpg", mask[0]*255)
                
                cv2.imwrite("masked_ref.jpg",X1i[0][0]*mask[0])
                yield [patch_input, coors, mask, X2i[0]/255.0, X1i[0]/255.0], None #Yield both images and their mutual label

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

        
    def load_data(self, test = False):
        if test:
            return next(self.test_generator)
        else:
            return next(self.train_generator)
    
    def load_dl(self):
        return [self.train_generator, self.test_generator]

'''
dl = Dataloader(train_path = "./data/train", test_path= "./data/test", batch_size = 1)
x, y = dl.load_data()
ps = x[0]
print(ps.shape)
p1 = x[0][:,:,:,:,:3]
p2 = x[0][:,:,:,:,3:]
print(p1.shape)
print(p2.shape)
visualize_corresponding_patches(p1,p2)

'''
# In[116]:


DIST_THRES = 100
RANDOM_PROB = 0.15
eps = 0.9
lr = 5e-4

def align_loss(ref_img, homo, mask, sns_img):
    aligned = tf.nn.convolution(sns_img, homo, padding = 'SAME')
    masked_aligned = aligned * mask
    masked_ref = ref_img * mask
    p_aligned = masked_aligned
    p_ref = masked_ref
    #p_aligned = VGG_bc4.predict(preprocess_input(masked_algined))
    #p_ref = VGG_bc4.predict(preprocess_input(masked_ref))
    print(p_aligned)
    loss = keras.losses.MSE(p_aligned, p_ref)
    return loss

def registration_loss(mask, sns_img, ref_img):
    def registration(dummy, homo):
        return align_loss(ref_img, homo, mask, sns_img,)
    return registration	


# In[123]:


def FeatureAlignNet(): 
    
    #similarity value for each match
    patches_plh = layers.Input(shape = ( BBOX_LENGTH-1, BBOX_LENGTH-1, IMG_CHN*2*NUM_MATCHES), dtype = 'float32', name = "input_patches" )
    
    #p = layers.Reshape((BBOX_LENGTH-1, BBOX_LENGTH-1, IMG_CHN*2*NUM_MATCHES))(patches_plh)
    p = layers.Conv2D(16, 3, strides=(2, 2), name = "pr_conv1",                                padding='valid', activation="relu", kernel_initializer='glorot_uniform')(patches_plh)
    p = layers.Conv2D(64, 3, strides=(2, 2), name = "pr_conv2",                                padding='valid', activation="relu", kernel_initializer='glorot_uniform')(p)
    p = layers.Conv2D(128, 3, strides=(1, 1), name = "pr_conv3",                                padding='valid', activation="relu", kernel_initializer='glorot_uniform')(p)
    p = layers.MaxPool2D()(p)
    p = layers.Flatten()(p)
    p = layers.Dense(16*NUM_MATCHES, activation= 'relu', name = "pr_fc1", kernel_initializer = 'glorot_uniform')(p)
    w = layers.Dense(NUM_MATCHES, activation= 'relu', name = "pr_out", kernel_initializer = 'glorot_uniform')(p)
    
    #homography calc
    coors_plh = layers.Input(shape = (NUM_MATCHES, 4), dtype = 'float32', name = "input_coors" )
    c = layers.Flatten()(coors_plh)
    cw = layers.Concatenate(axis=-1)([c,w])
    #h = layers.Dense(256, activation= 'relu', name = "homo_fc1", kernel_initializer = 'glorot_uniform')(cw)
    h = layers.Dense(27, activation= 'relu', name = "homo_out", kernel_initializer = 'glorot_uniform')(cw)
    homo = layers.Reshape((3,3,3))(h)
    
    mask_plh = keras.Input(shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHN),                               dtype = 'float32', name = "input_mask" )
    sns_plh = keras.Input(shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHN),                               dtype = 'float32', name = "input_sns" )
    ref_plh = keras.Input(shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHN),                               dtype = 'float32', name = "input_ref" )
    
    model = keras.Model(inputs=[patches_plh, coors_plh, mask_plh, sns_plh, ref_plh], outputs=homo)
    loss_func = registration_loss(mask_plh, sns_plh, ref_plh)
    model.compile(optimizer = keras.optimizers.Adam(learning_rate=lr),
              loss=loss_func)
    
    print(model.summary())
    return model


# In[124]:


batch_size = 6
trainset_size = 6
testset_size = 6
epochs = 1000


dl = Dataloader(train_path = "./data/train", test_path= "./data/test", batch_size = batch_size)
train_generator, test_generator = dl.load_dl()
net = FeatureAlignNet()
print("Trainables:")
print(tf.trainable_variables())
net.fit_generator(train_generator,
                                steps_per_epoch=trainset_size/batch_size,
                                epochs = epochs,
                                validation_data = test_generator,
                                validation_steps = testset_size/batch_size,
                                use_multiprocessing = True,
                                shuffle = True)


# In[ ]:


patches, additionals = dl.load_data(test=True)
pred = net(patches)
refs,snss,matches = additionals
imgs = register_with_predicted(pred, matches, refs, snss)
for i in range(imgs.shape[0]):
    cv2.imwrite("registered_cv2{}.jpg".format(i), imgs[i])


# In[ ]:







