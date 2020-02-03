import numpy as np
import cv2

NUM_F_POINTS = 2000
IMG_WIDTH = 620
IMG_HEIGHT = 877
IMG_CHN = 3
NUM_F_POINTS = 5000
NUM_MATCHES = 500
BBOX_LENGTH = 21  

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
    
    
def test_cv2_batch(dl):
    refs, sns = dl.load_image()
    kp1s, kp2s, d1s,d2s = extract_feature_batch(refs,sns)
    matches = batch_match(d1s, d2s)
    
    visualize_matches(refs, sns, kp1s, kp2s, matches)
    homos = calc_homographies(kp1s, kp2s, matches)
    imgs = register_images(sns, homos)
    for i in range(6):
            cv2.imwrite("registered_cv2{}.jpg".format(i), imgs[i])
        


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


def extract_match_patches(ref_imgs, sns_imgs, kprs, kpss, drs, dss):
    '''
    output: N * NUM_MATCHES * 2 * PATCH_H * PATCH_W * CHN Example:(6, 500, 2, 6, 6, 3)
    '''
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    patches, matches = [], []
    for i in range(kprs.shape[0]):
        results = matcher.match(drs[i], dss[i])
        patch = []
        match = []
        for r in results:
            pair = []
            match_p = []
            if len(patch) >= NUM_MATCHES: break
            x, y, dis = kprs[i][r.queryIdx].astype(np.uint32)
            if  x-(BBOX_LENGTH-1) > 0 and y-(BBOX_LENGTH-1) > 0 \
                and x+(BBOX_LENGTH-1)/2 < IMG_WIDTH and y+(BBOX_LENGTH-1)/2 < IMG_HEIGHT:
                pair.append(ref_imgs[i][y-int((BBOX_LENGTH-1)/2):y+int((BBOX_LENGTH-1)/2)\
                                       ,x-int((BBOX_LENGTH-1)/2):x+int((BBOX_LENGTH-1)/2)])
                
            x, y, dis = kpss[i][r.trainIdx].astype(np.uint32)
            if  x-(BBOX_LENGTH-1) > 0 and y-(BBOX_LENGTH-1) > 0 \
                and x+(BBOX_LENGTH-1)/2 < IMG_WIDTH and y+(BBOX_LENGTH-1)/2 < IMG_HEIGHT:
                pair.append(sns_imgs[i][y-int((BBOX_LENGTH-1)/2):y+int((BBOX_LENGTH-1)/2)\
                                       ,x-int((BBOX_LENGTH-1)/2):x+int((BBOX_LENGTH-1)/2)])
                
            if len(pair) == 2:
                patch.append(np.array(pair))
                match.append(r)
                
        while len(patch) < NUM_MATCHES:
            patch.append(np.zeros((2,BBOX_LENGTH-1,BBOX_LENGTH-1, IMG_CHN)))
            match.append(None)
            
        patch = np.array(patch)
        patches.append(patch)
        match = np.array(match)
        matches.append(match)

    patches = np.array(patches)
    matches = np.array(matches)   

    return [patches[:,:,0,:,:,:], patches[:,:,1,:,:,:], matches]
    

def get_match_info(refs,sns):
    """
    returns in 255 scale
    """
    kprs,kpss, drs, dss = extract_feature_coor_batch(refs,sns)
    p_ref, p_sns, matches =  extract_match_patches(refs, sns, kprs, kpss, drs, dss)
    return [p_ref, p_sns, matches, kprs, kpss]
    
    
def get_model_inputs(refs,sns):
    p_ref, p_sns, matches, kprs, kpss = get_match_info(refs, sns)
    p_ref = (p_ref.astype(np.float32) / 255.0 ).reshape((p_ref.shape[0]*p_ref.shape[1],\
                                                        p_ref.shape[2], p_ref.shape[3], p_ref.shape[4]))
    p_sns = (p_sns.astype(np.float32) / 255.0).reshape((p_sns.shape[0]*p_sns.shape[1],\
                                                        p_sns.shape[2], p_sns.shape[3], p_sns.shape[4]))
    #matches = matches.reshape(matches.shape[0] * matches.shape[1])
    return [p_ref, p_sns, matches, kprs, kpss]
        
def visualize_corresponding_patches(p1, p2):
    for j in range(50):
        vis = (np.concatenate((p1[0][j], p2[0][j]), axis=1))
        cv2.imwrite("patch pair {}.jpg".format(j), vis)
        
def visualize_coords(img, c):
    for j in range(500):
        cv2.circle(img[0], (c[0][j][0], c[0][j][1]) , 1, (0, 0, 255), -1)
    cv2.imwrite("New feture img.jpg", img[0])


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
        
