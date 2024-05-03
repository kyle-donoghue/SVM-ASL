import numpy as np
from sklearn.metrics import f1_score
import cv2

LABEL_DEFS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
TOTAL_IMAGES = 3000

def import_images(num_training, num_testing, data_path):
    num_labels = len(LABEL_DEFS)
    training_images = np.zeros((num_training*num_labels, 200, 200))
    training_labels = np.zeros((num_training*num_labels))
    testing_images = np.zeros((num_testing*num_labels, 200, 200))
    testing_labels = np.zeros((num_testing*num_labels))

    shuffled = np.random.permutation(TOTAL_IMAGES)

    training_i = shuffled[:num_training]
    testing_i = shuffled[num_training:num_training+num_testing]

    count = 0
    for i in range(num_labels):
        filename = data_path+"asl_alphabet_train\\asl_alphabet_train\\"
        for p in training_i:
            filename_tmp = filename + LABEL_DEFS[i]+"\\"+LABEL_DEFS[i]+str(p+1)+".jpg"
            training_images[count] = cv2.imread(filename_tmp, cv2.IMREAD_GRAYSCALE)
            training_labels[count] = i
            print('Loading Training: '+str(np.round(count/len(training_images)*100))+"%", end='\r', flush=True)
            count += 1
    print()

    count = 0
    for p in testing_i:
        filename = data_path+"asl_alphabet_train\\asl_alphabet_train\\" # use extra training data for testing
        for i in range(num_labels):
            filename_tmp = filename + LABEL_DEFS[i]+"\\"+LABEL_DEFS[i]+str(p+1)+".jpg"
            testing_images[count] = cv2.imread(filename_tmp, cv2.IMREAD_GRAYSCALE)
            testing_labels[count] = i
            print('Loading Training: '+str(np.round(count/len(testing_images)*100))+"%", end='\r', flush=True)
            count += 1
    print()
    
    return training_images.astype('uint8'), training_labels, testing_images.astype('uint8'), testing_labels

def shuffle_split(num_training, num_testing,num_labels=29):
    shuffled = np.random.permutation(TOTAL_IMAGES)

    label_training_i = shuffled[:num_training]
    label_testing_i = shuffled[num_training:num_training+num_testing]

    training_i = []
    testing_i = []

    for i in range(num_labels):
        training_i.extend(label_training_i + i*TOTAL_IMAGES)
        testing_i.extend(label_testing_i + i*TOTAL_IMAGES)

    return training_i, testing_i


def extract_hog_features(hog, images):
    hog_features = []
    for count, img in enumerate(images):
        print('Extracting HOG: '+str(np.round(count/len(images)*100))+"%", end='\r', flush=True)
        hog_features.append(hog.compute(img))
    print()
    return np.array(hog_features)

def calculate_hog_feature_length(param):
    return int((param['win_size'][0] - param['block_size'][0]) / param['block_stride'][0] + 1) * int((param['win_size'][1] - param['block_size'][1]) / param['block_stride'][1] + 1) * param['n_bins'] * int(param['block_size'][0] / param['cell_size'][0]) * int(param['block_size'][1] / param['cell_size'][1])

def add_noise(images, snr):
    noisy_images = np.zeros_like(images)
    linear_snr = 10**(snr/10)
    for i in range(len(images)):
        avg_power = np.mean(images[i]**2)
        noise_power = avg_power/linear_snr
        noise = np.random.normal(0, np.sqrt(noise_power), images[i].shape)
        img_tmp = images[i].astype('float64')+noise
        img_tmp = np.clip(img_tmp, 0, 255)
        noisy_images[i] = img_tmp.astype('uint8')
    return noisy_images

def add_variable_noise(images, snr_min, snr_max):
    noisy_images = np.zeros_like(images)
    for i in range(len(images)):
        snr = np.random.uniform(snr_min, snr_max)
        linear_snr = 10**(snr/10)
        avg_power = np.mean(images[i]**2)
        noise_power = avg_power/linear_snr
        noise = np.random.normal(0, np.sqrt(noise_power), images[i].shape)
        img_tmp = images[i].astype('float64')+noise
        img_tmp = np.clip(img_tmp, 0, 255)
        noisy_images[i] = img_tmp.astype('uint8')
    return noisy_images

def find_min_f1(pred,truth):
    f1 = f1_score(truth,pred,average=None)
    min_f1 = np.min(f1)
    min_ind = np.argmin(f1)
    min_label = LABEL_DEFS[min_ind]
    return min_f1, min_label