"""
Create a feature vector for each image using the local binary pattern (LBP) method.
"""

import numpy as np
from skimage import feature as skimage_feature

import prepare_features


def extract_lbp_features(base_path=None, output_path='./img.w.lbp.features.tab'):

    """
    Extract LBP features and save them to disk.
    :param base_path: Path that contains the download summary and the photo directory
    :return: Dataframe that contains LBP features for each of the downloaded images
    """

    df = prepare_features.load_descriptions(base_path)
    df['lbp_feature'] = None

    total_images = df.shape[0]
    images_processed = 0
    for idx, row in df.iterrows():
        # Provide a status update if necessary
        if images_processed % 100.0 == 0:
            print(str(images_processed) + '/' + str(total_images) + ' images (i.e. ' +
                  str(np.round(images_processed / total_images, 3) * 100) + "%) complete.")

        # Extract LBP features for image
        img = prepare_features.read_grayscale_img(row.img_path)
        df.at[idx, 'lbp_feature'] = lbp(img)
        images_processed += 1

    # Write the dataframe to disk
    if output_path is not None:
        df.to_csv(output_path, index=False, sep="\t")

    return df


def lbp(x, p=8, r=1):

    """
    Create a histogram containing the local binary pattern features of each image.
    :param x: Input image vector (default: 8)
    :param p: Number of surrounding points to use per-pixel (default: 1)
    :param r: Radius of circle from which "surrounding points" are selected
    :return: Histogram of LBP features
    """

    features = np.histogram(skimage_feature.local_binary_pattern(x, P=p, R=r).ravel(),
                            bins=2 ** p, range=(0, 2 ** p), normed=True)[0].tolist()
    return features


if __name__ == '__main__':

    extract_lbp_features()

