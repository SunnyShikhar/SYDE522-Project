import json
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

    return features[0]


def extract_deep_features(category, base_path=None, output_file_name='img.w.deep.features'):

    # Write to the category-specific directory
    if base_path is None:
        base_path = ''
    output_path = base_path + './deep/' + category + '/' + output_file_name

    # Backup dict to store the deep features by flickr photo ID just in case
    backup_dict = {}

    model = applications.resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')

    df = prepare_features.load_descriptions(base_path)
    df = df[df.category == category]
    df['deep_feature'] = None

    total_images = df.shape[0]
    images_processed = 0
    flag = 1
    for idx, row in df.iterrows():
        # Provide a status update if necessary
        if images_processed % 100.0 == 0:
            print(str(images_processed) + '/' + str(total_images) + ' images (i.e. ' +
                  str(np.round(images_processed / total_images, 3) * 100) + "%) complete.")
            if flag == 1:
                flag = 2
            else if flag == 2:
                flag = 1
            # Save a temporary file
            temp_name = output_path + '.temp.' + str(flag) + '.complete'
            df.to_pickle(temp_name + '.p')
            # Write to a JSON file just in case as well
            with open(temp_name + '.json', 'w') as fp:
                json.dump(backup_dict, fp)

        # Extract Deep Features
        img = prepare_features.read_colour_img(row.img_path)
        try:
            feature = get_deep_feature(model, img)
            df.at[idx, 'deep_feature'] = feature
            backup_dict[row.flickr_photo_id] = feature.tolist()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            df.at[idx, 'deep_feature'] = None
            backup_dict[row.flickr_photo_id] = None

        images_processed += 1

    # Also pickle the dataframe
    df.to_pickle(output_path + '.p')
    # Also write the backup dictionary to a JSON file
    with open(output_path + '.json', 'w') as fp:
        json.dump(backup_dict, fp)

    return df

if __name__ == '__main__':

    extract_deep_features(category='urban')