from tensorflow import keras
from utils import *

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
                X1i = gen_ref.next()
                X2i = gen_sns.next()
                x,y,matches, _, _ = get_model_inputs(X1i[0].astype(np.uint8), X2i[0].astype(np.uint8))
                
                yield [ X1i[0].astype(np.uint8), X2i[0].astype(np.uint8) ], None #Yield both images and their mutual label
                
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

        
    def load_image(self, test = False):
        if test:
            return next(self.test_generator)[0]
        else:
            return next(self.train_generator)[0]
    
    def load_dl(self):
        return [self.train_generator, self.test_generator]
