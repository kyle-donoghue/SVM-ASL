import processing
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pathlib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

NUM_TRAINING = 0
NUM_TESTING = 50

# kaggle dataset should be extracted into this folder
DIR_PATH = pathlib.Path(__file__).parent.parent.resolve()

DATA_PATH = str(DIR_PATH) + "\\dataset\\"
MODEL_PATH = str(DIR_PATH) + "\\model\\"


if __name__ == '__main__':

    param = {
        'gamma_correction': True,
        'win_size': (200, 200),
        'n_bins': 9,
        'cell_size': (40, 40),
        'block_size': (80, 80),
        'block_stride': (40, 40)
    }
    hog = cv2.HOGDescriptor(param['win_size'], param['block_size'], param['block_stride'], param['cell_size'], param['n_bins'])
    print("HOG Feature Length: ", processing.calculate_hog_feature_length(param))

    # import all images and labels from the .pkl files for faster loading
    print("Loading Images... ", end="")
    all_images = joblib.load(DATA_PATH+'all_images.pkl')
    all_labels = joblib.load(DATA_PATH+'all_labels.pkl')
    print("Done")

    indice_split = processing.shuffle_split(NUM_TRAINING, NUM_TESTING,2) # shuffle and split the data into training and testing sets
    training_images = all_images[indice_split[0]]
    training_labels = all_labels[indice_split[0]]
    testing_images = all_images[indice_split[1]]
    testing_labels = all_labels[indice_split[1]]
    

    testing_features = processing.extract_hog_features(hog, testing_images)

    normalized_testing_features = StandardScaler().fit_transform(testing_features)
    pca = PCA(n_components=20)
    pca.fit(normalized_testing_features)

    print("PCA Explained Variance: ", pca.explained_variance_ratio_)

    pca_testing_features = pca.transform(normalized_testing_features)

    print("PCA Testing Features Shape: ", pca_testing_features.shape)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for i in range(len(pca_testing_features)):
        if testing_labels[i] == 1:
            ax.scatter(pca_testing_features[i][0], pca_testing_features[i][1], c='r')
        else:
            ax.scatter(pca_testing_features[i][0], pca_testing_features[i][1], c='b')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title('PCA Analysis of Testing Data')
    ax.grid()

    plt.show()



