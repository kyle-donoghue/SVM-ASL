import processing
import joblib
from sklearn import pipeline, preprocessing, model_selection
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pathlib


# kaggle dataset should be extracted into this folder
DIR_PATH = pathlib.Path(__file__).parent.parent.resolve()

DATA_PATH = str(DIR_PATH) + "\\dataset\\"
MODEL_PATH = str(DIR_PATH) + "\\model\\"


MODELS = ["GOOD_","10snr_","0_40snr_"]
model_names = ["Trained with Original", "Trained with 10 SNR", "Trained with 0-40 SNR"]

if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for i,MODEL_ID in enumerate(MODELS):
        scores = joblib.load(MODEL_PATH+'hog\\'+MODEL_ID+'snr_scores.pkl')
        snrs = joblib.load(MODEL_PATH+'hog\\'+MODEL_ID+'snr_snrs.pkl')
        ax.plot(snrs,scores,linewidth=2,label=model_names[i],color='C'+str(i+2))


    
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Testing Accuracy')
    ax.set_title('Accuracy vs SNR')
    # ax.set_xticks(TRAINING_SIZE)
    ax.grid()
    ax.legend()

    plt.show()
