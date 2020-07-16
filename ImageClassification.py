
from keras.applications.imagenet_utils import preprocess_input,decode_predictions
from keras.models import load_model
from keras.preprocessing import image

from keras.applications.vgg19 import VGG19

model=VGG19(weights="imagenet")
model.save('vgg19.h5')


