import csv
import os

from keras import applications
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
import numpy as np

def get_features(id, directory, eachFile):
    model = applications.resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')
    print (eachFile)
    img = image.load_img(directory + eachFile)

    # convert image to numpy array
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # extract the features
    features = model.predict(x)[0]

    # convert from numpy to a list of values
    features_arr = np.char.mod('%f', features)

    # write to tsv file
    with open('records.tsv', 'a') as tsvfile:
        w = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        w.writerows([id, features_arr])

    return (features_arr)

# Image Path directory
strDir = '../photos/'

# Iterate through all images in photos folder
for file in os.listdir(os.fsencode(strDir)):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"):
        id = filename.split('.')[0]
        featList = get_features(id, strDir, filename)
        print(featList)

    else:
        print ('no file found')