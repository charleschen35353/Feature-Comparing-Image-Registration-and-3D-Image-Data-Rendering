import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from utils import *

RANDOM_PROB = 0.15
eps = 0.9

def feature_extractor_layers(): # shared layer
    feature_extractor_input = keras.Input(shape=( (2*int((BBOX_LENGTH-1)/2))**2*IMG_CHN, ) )
    x = layers.Flatten()(feature_extractor_input)
    x = layers.Dense(128 ,activation='relu', name = "fc1", kernel_initializer= keras.initializers.glorot_normal())(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu', name= "fc2",kernel_initializer= keras.initializers.glorot_normal())(x)
    x = layers.BatchNormalization()(x)
    feature_extractor_output = layers.Dropout(0.2)(x)
    feature_extractor = keras.models.Model(feature_extractor_input,feature_extractor_output, name='feature_extractor')
    
    return feature_extractor

class PatchRanker(keras.Model):
    def __init__(self, name = "patch_ranker", **kwargs):
        super(PatchRanker, self).__init__(name=name, **kwargs)
        self.feature_extractor = feature_extractor_layers()
        self.iteration = 0
        
    def call(self, inputs):
        refs, snss = inputs
        refs = refs.numpy()
        snss = snss.numpy()
        x,y,matches, _ ,_ = get_model_inputs(refs,snss)
        
        with tf.GradientTape() as gtape:
            f_x = self.feature_extractor(x)
            f_y = self.feature_extractor(y)
            f_xy = tf.concat([f_x,f_y],1)
            z = layers.Dense(64 ,activation='relu', name = "fc3" ,kernel_initializer = keras.initializers.glorot_normal())(f_xy)
            z = layers.BatchNormalization()(z)
            z = layers.Dropout(0.2)(z)
            classified = layers.Dense(1 , activation= 'relu',kernel_initializer=keras.initializers.glorot_normal())(z)
            pred = tf.reshape(classified, [-1, NUM_MATCHES]) 
            inds = tf.argsort(pred, axis = 1)
            
            #prevent overly-greedy 
            if tf.random.uniform([1], dtype=tf.dtypes.float32) < RANDOM_PROB * eps**(self.iteration / 100):
                inds = tf.random.shuffle(inds)
            mask = tf.cast(tf.where(inds < 50, tf.ones_like(inds), tf.zeros_like(inds)), tf.float32 ) # -1*NUM_MATCHES*1
            selected_matches = np.where( mask >= 0.5, matches, np.zeros_like(matches)  )
            good_matches = []
            for i in range(selected_matches.shape[0]):
                good_matches.append(selected_matches[i][selected_matches[i] != 0])

            kprs, kpss, _, _ =  extract_feature_batch(refs,snss)
            homo = calc_homographies(kprs, kpss, good_matches)
            imgs = np.array(register_images(snss, homo))

            #for i in range(imgs.shape[0]):
                #tf.summary.image('reg_img{}'.format(i), imgs)
                #cv2.imwrite("registered_cv2{}.jpg".format(i), imgs[i])
            #print("registered!!")
            _, _, matches, kp1s, kp2s = get_match_info(refs, imgs)

            feature_diss = []
            coor_diss = []

            for i in range(kp1s.shape[0]):
                coor_dis = 0
                feature_dis = 0
                valid_count = 0
                for r in matches[i]:	
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


            gt = []
            for c_d, f_d in zip(coor_diss, feature_diss):
                gt.append(tf.ones_like(pred[0]) * 1/(c_d+f_d) )
                print(c_d)
                print(f_d)
                print("================")
            gt = np.array(gt)

            loss = tf.reduce_mean(((gt - pred)*mask)**2)
            self.add_loss(loss)
            self.iteration += 1
        '''    
        grads = gtape.gradient(pred, self.trainable_variables)
        
        for i in range(len(grads)):
            print("Gradietns:")
            print(grads[i])
            print("Weights:")
            print(self.trainable_variables[i])
            print("==========================")
        '''
        return imgs

