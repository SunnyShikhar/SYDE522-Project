from keras import applications
from keras.applications.resnet50 import preprocess_input
import numpy as np
from skimage import feature as skimage_feature

import prepare_features

### Local Binary Patterns ###

def extract_lbp_features(base_path=None, output_path='./img.w.lbp.features.tab'):

    """Iterate through the dataset, read in each image, and create the LBP feature vector."""

    # Load the dataset from the 'base_path'.
    df = prepare_features.load_descriptions(base_path)
    df['lbp_feature'] = None

    for idx, row in df.iterrows():
        # Extract LBP features for image
        img = prepare_features.read_grayscale_img(row.img_path)
        df.at[idx, 'lbp_feature'] = lbp(img)

    # Write the features to a CSV
    if output_path is not None:
        df.to_csv(output_path, index=False, sep="\t")

def lbp(x, p=8, r=1):

    """Generate an LBP Feature vector for the input image 'x'."""

    features = np.histogram(skimage_feature.local_binary_pattern(x, P=p, R=r).ravel(),
                            bins=2 ** p, range=(0, 2 ** p), normed=True)[0].tolist()
    return features

### Deep Features ###

def get_deep_feature(model, img):

    """Generate a deep feature vector for the input image using the input model."""

    # prepare the image for the CNN
    img = img.astype(np.float64)
    img = np.expand_dims(img, axis=0)
    preprocessed_img = preprocess_input(img)

    # extract the deep features
    features = model.predict(preprocessed_img)
    return features[0]

def extract_deep_features(output_path, base_path=None):

    """Iterate through the dataset, read in each image, and create the deep feature vector."""

    # Load the pre-trained CNN.
    model = applications.resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')
    # model = applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')

    df = prepare_features.load_descriptions(base_path)
    df['deep_feature'] = None

    for idx, row in df.iterrows():
        # Extract Deep Features
        img = prepare_features.read_colour_img(row.img_path)
        # Try-except is necessary because images that are inappropriately sized for the CNN will fail.
        try:
            feature = get_deep_feature(model, img)
            df.at[idx, 'deep_feature'] = feature
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            df.at[idx, 'deep_feature'] = None

    # Pickle the dataframe
    df.to_pickle(output_path + '.p')