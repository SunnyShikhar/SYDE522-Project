from keras import applications
from keras.applications.resnet50 import preprocess_input
import numpy as np

import prepare_features


def get_deep_feature(model, img):

    # convert image to numpy array
    img = img.astype(np.float64)
    img = np.expand_dims(img, axis=0)
    preprocessed_img = preprocess_input(img)

    # extract the features
    features = model.predict(preprocessed_img)

    return features


def extract_deep_features(base_path=None, output_path='./img.w.deep.features.tab'):

    model = applications.resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')

    df = prepare_features.load_descriptions(base_path)
    df['deep_feature'] = None

    total_images = df.shape[0]
    images_processed = 0
    for idx, row in df.iterrows():
        # Provide a status update if necessary
        if images_processed % 100.0 == 0:
            print(str(images_processed) + '/' + str(total_images) + ' images (i.e. ' +
                  str(np.round(images_processed / total_images, 3) * 100) + "%) complete.")

        # Extract LBP features for image
        img = prepare_features.read_colour_img(row.img_path)
        df.at[idx, 'deep_feature'] = get_deep_feature(model, img)
        images_processed += 1

    # Write the dataframe to disk
    if output_path is not None:
        df.to_csv(output_path, index=False, sep="\t")

    return df


if __name__ == '__main__':

    extract_deep_features()