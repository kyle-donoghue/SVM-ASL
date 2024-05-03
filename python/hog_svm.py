import processing
import joblib
from sklearn import pipeline, preprocessing, model_selection
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pathlib

# cpu only
from sklearnex import patch_sklearn 
patch_sklearn()
from sklearn.svm import SVC


NUM_TRAINING = 2250
NUM_TESTING = 750
EXPORT_MODEL = False
USE_EXISTING_MODEL = True
ADD_NOISE = False
ADD_VARIABLE_NOISE = False
SNR = 10
SNR_MIN = 0
SNR_MAX = 40
USE_PCA = False
PCA_MIN_VARIANCE = .8
MODEL_ID = "GOOD_"

DIR_PATH = pathlib.Path(__file__).parent.parent.resolve()

DATA_PATH = str(DIR_PATH) + "\\dataset\\"
MODEL_PATH = str(DIR_PATH) + "\\model\\"

def setup(existing):
    if existing:
        param = joblib.load(MODEL_PATH+'hog\\'+MODEL_ID+'hog_param.pkl') # load the hog parameters
        pipeline_model = joblib.load(MODEL_PATH+'hog\\'+MODEL_ID+'hog_model.pkl') # load the SVM model
        indice_split = joblib.load(MODEL_PATH+'hog\\'+MODEL_ID+'hog_indice.pkl') # load the indices for the training and testing sets
    else:
        param = {
            'gamma_correction': True,
            'win_size': (200, 200),
            'n_bins': 9,
            'cell_size': (40, 40),
            'block_size': (80, 80),
            'block_stride': (40, 40)
        }
        param_grid={'C':[0.1,1,10,100], 
                'gamma':[0.0001,0.001,0.1,1], 
                'kernel':['rbf','poly']}
        svc = SVC(probability=False,verbose=0) # create SVM model
        svm_model = model_selection.GridSearchCV(svc, param_grid,verbose=10) # create grid search
        pipeline_model = pipeline.make_pipeline(preprocessing.StandardScaler(), svm_model) # create pipeline to normalize data and train model
        indice_split = processing.shuffle_split(NUM_TRAINING, NUM_TESTING) # shuffle and split the data into training and testing sets
    return param, indice_split, pipeline_model

def test_model(model, testing_images, testing_labels):
    print("Using parameters: ", model._final_estimator.best_params_)
    print("with score: ", model._final_estimator.best_score_)
    testing_features = processing.extract_hog_features(hog, testing_images)
    if USE_PCA:
        pca = joblib.load(MODEL_PATH+'hog\\'+'pca_model.pkl')
        testing_features = pca.transform(testing_features)
        variance = pca.explained_variance_ratio_
        pca_components = np.argmax(np.cumsum(variance) > PCA_MIN_VARIANCE)
        testing_features = testing_features[:,:pca_components]
        print("PCA Testing Features Shape: ", testing_features.shape)
    print("Testing Model... ", end="", flush=True)
    predictions = model.predict(testing_features)
    print("Done")
    print("Accuracy: ", accuracy_score(testing_labels, predictions))
    f1_scores = f1_score(testing_labels, predictions, average='macro')
    min_f1, min_label = processing.find_min_f1(predictions, testing_labels)
    print("Minimum F1: ", min_f1, " at label: ", min_label)
    cm = confusion_matrix(testing_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=processing.LABEL_DEFS)
    disp.plot()
    plt.show()

def train_model(model, training_images, training_labels, export):
    if ADD_NOISE:
        training_images_noisy = processing.add_noise(training_images, SNR)
        training_features = processing.extract_hog_features(hog, training_images_noisy)
    elif ADD_VARIABLE_NOISE:
        training_images_noisy = processing.add_variable_noise(training_images, SNR_MIN, SNR_MAX)
        training_features = processing.extract_hog_features(hog, training_images_noisy)
    else:
        training_features = processing.extract_hog_features(hog, training_images)
    if USE_PCA:
        pca = joblib.load(MODEL_PATH+'hog\\'+'pca_model.pkl')
        training_features = pca.transform(training_features)
        variance = pca.explained_variance_ratio_
        pca_components = np.argmax(np.cumsum(variance) > PCA_MIN_VARIANCE)
        training_features = training_features[:,:pca_components]
        print("PCA Training Features Shape: ", training_features.shape)
    model.fit(training_features, training_labels)
    print("Best Parameters: ", model._final_estimator.best_params_)
    print("Best Score: ", model._final_estimator.best_score_)
    if export:
        joblib.dump(model, MODEL_PATH+'hog\\'+MODEL_ID+'hog_model.pkl')
        joblib.dump(param, MODEL_PATH+'hog\\'+MODEL_ID+'hog_param.pkl')
        joblib.dump(indice_split, MODEL_PATH+'hog\\'+MODEL_ID+'hog_indice.pkl')
        print("Model Exported")

def show_validation_curves(model):
    est = model._final_estimator

    scores = est.cv_results_['mean_test_score']
    poly_scores = est.cv_results_['mean_test_score'][est.cv_results_['param_kernel'] == 'poly']
    rbf_scores = est.cv_results_['mean_test_score'][est.cv_results_['param_kernel'] == 'rbf']
    
    C = est.param_grid['C']
    gamma = est.param_grid['gamma']
    
    poly_grid = np.zeros([len(C), len(gamma)])
    rbf_grid = np.zeros([len(C), len(gamma)])

    for i,c in enumerate(C):
        for j,g in enumerate(gamma):
            poly_grid[i][j] = scores[np.logical_and((np.logical_and(est.cv_results_['param_C']==c, est.cv_results_['param_gamma']==g)),(est.cv_results_['param_kernel']=='poly'))].item()
            rbf_grid[i][j] = scores[np.logical_and((np.logical_and(est.cv_results_['param_C']==c, est.cv_results_['param_gamma']==g)),(est.cv_results_['param_kernel']=='rbf'))].item()

    X,Y = np.meshgrid(np.log10(gamma), np.log10(C))

    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(X, Y, poly_grid, cmap='viridis')
    ax.set_title('Poly Scores')
    ax.set_xlabel('log10(gamma)')
    ax.set_ylabel('log10(C)')
    ax.set_zlabel('Score')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(X, Y, rbf_grid, cmap='viridis')
    ax.set_title('RBF Scores')
    ax.set_xlabel('log10(gamma)')
    ax.set_ylabel('log10(C)')
    ax.set_zlabel('Score')
    plt.show()



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

    if USE_EXISTING_MODEL:
        print("Model Loaded")
    else:
        train_model(pipeline_model, training_images, training_labels, EXPORT_MODEL)
    
    show_validation_curves(pipeline_model)

    test_model(pipeline_model, testing_images, testing_labels)

