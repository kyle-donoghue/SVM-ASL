import processing
import joblib
from sklearn import pipeline, preprocessing, model_selection
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pathlib

NUM_TESTING = 100
EXPORT_MODEL = False
USE_EXISTING_MODEL = True
MODEL_ID = "GOOD_"

# kaggle dataset should be extracted into this folder
DIR_PATH = pathlib.Path(__file__).parent.parent.resolve()

DATA_PATH = str(DIR_PATH) + "\\dataset\\"
MODEL_PATH = str(DIR_PATH) + "\\model\\"


def setup(existing):
    if existing:
        param = joblib.load(MODEL_PATH+'hog\\'+MODEL_ID+'hog_param.pkl') # load the hog parameters
        pipeline_model = joblib.load(MODEL_PATH+'hog\\'+MODEL_ID+'hog_model.pkl') # load the SVM model
        indice_split = joblib.load(MODEL_PATH+'hog\\'+MODEL_ID+'hog_indice.pkl') # load the indices for the training and testing sets
        indice_split = processing.shuffle_split(0, NUM_TESTING) 
    else:
        print("oops")
    return param, indice_split, pipeline_model


if __name__ == '__main__':

    param, indice_split, pipeline_model = setup(USE_EXISTING_MODEL)
    hog = cv2.HOGDescriptor(param['win_size'], param['block_size'], param['block_stride'], param['cell_size'], param['n_bins'])
    print("HOG Feature Length: ", processing.calculate_hog_feature_length(param))

    # import all images and labels from the .pkl files for faster loading
    print("Loading Images... ", end="")
    all_images = joblib.load(DATA_PATH+'all_images.pkl')
    all_labels = joblib.load(DATA_PATH+'all_labels.pkl')

    training_images = all_images[indice_split[0]]
    training_labels = all_labels[indice_split[0]]
    testing_images = all_images[indice_split[1]]
    testing_labels = all_labels[indice_split[1]]
    print("Done")


    scores = []
    snrs = np.linspace(-10, 40, 21)

    for snr in snrs:
        print("SNR: ", snr)
        testing_images_noisy = processing.add_noise(testing_images, snr)
        testing_features = processing.extract_hog_features(hog, testing_images_noisy)
        predictions = pipeline_model.predict(testing_features)
        score = accuracy_score(testing_labels, predictions)
        print("Accuracy: ", score)
        print()
        scores.append(score)

    # save scores and snrs in pkl
    joblib.dump(scores, MODEL_PATH+'hog\\'+MODEL_ID+'snr_scores.pkl')
    joblib.dump(snrs, MODEL_PATH+'hog\\'+MODEL_ID+'snr_snrs.pkl')

    