import processing
import joblib
import pathlib

# kaggle dataset should be extracted into this folder
DIR_PATH = pathlib.Path(__file__).parent.parent.resolve()

DATA_PATH = str(DIR_PATH) + "\\dataset\\"
MODEL_PATH = str(DIR_PATH) + "\\model\\"

all_images, all_labels, _, _ = processing.import_images(3000, 0, DATA_PATH)

joblib.dump(all_images, DATA_PATH+'all_images.pkl')
joblib.dump(all_labels, DATA_PATH+'all_labels.pkl')
print("Images Exported")