import processing
import joblib
from sklearn import pipeline, preprocessing, model_selection
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pathlib

MODEL_PREFIXES = ["10samp_","25samp_","50samp_","100samp_", "250samp_", "500samp_", "1000samp_", "GOOD_"]
TRAINING_SIZE = [10,25,50,100,250,500,1000,2250]

# kaggle dataset should be extracted into this folder
DIR_PATH = pathlib.Path(__file__).parent.parent.resolve()

DATA_PATH = str(DIR_PATH) + "\\dataset\\"
MODEL_PATH = str(DIR_PATH) + "\\model\\"


C_FIXED = .1
GAMMA_FIXED = .1

def extract_validation_score(model,C,gamma):
    est = model._final_estimator
    poly_score = est.cv_results_['mean_test_score'][np.logical_and(np.logical_and(est.cv_results_['param_kernel']=='poly',est.cv_results_['param_C']==C),est.cv_results_['param_gamma']==gamma)]
    return poly_score

def load_model(prefix):
    pipeline_model = joblib.load(MODEL_PATH+'hog\\'+prefix+'hog_model.pkl') # load the SVM model
    return pipeline_model

scores = []
for prefix in MODEL_PREFIXES:
    print("Model: ", prefix)
    model = load_model(prefix)
    score = extract_validation_score(model,C_FIXED,GAMMA_FIXED)
    scores.append(score)
    print("Validation Score: ", score)


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.plot(TRAINING_SIZE,scores,marker='o')
ax.set_xlabel('Training Size')
ax.set_ylabel('Validation Score')
ax.set_title('5-fold Validation Score vs Training Size')
# ax.set_xticks(TRAINING_SIZE)
ax.grid()

plt.show()