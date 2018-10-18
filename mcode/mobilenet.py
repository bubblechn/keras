# coding: utf-8
from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input,decode_predictions
from keras.models import Model
import numpy as np
import time

model = MobileNet(weights='imagenet', include_top=True,classes=1000)

start = time.time()

img_path = '../mdataset/img_test/dog.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=15)[0])
end = time.time()

print('time:\n')
print str(end-start)


