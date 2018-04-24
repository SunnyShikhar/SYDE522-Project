import numpy as np
import pandas as pd
from scipy.misc import imread


def load_descriptions(base_path=None):

    """Load the CrowdFlower features DataFrame."""

    if base_path is None: base_path = ''

    df = pd.read_csv(base_path + './download_summary.tsv', sep="\t", index_col=False)

    df = df[df.downloaded == True]

    df = df.rename(columns={'#flickr_photo_id': 'flickr_photo_id'})
    df['img_path'] = df.flickr_photo_id.apply(
        lambda x: base_path + './photos/' + str(x) + '.jpg'
    )

    return df


def read_colour_img(path):

    """Read colour image from disk."""

    return imread(path, mode='RGB')


def read_grayscale_img(path):

    """Read gray-scale image from disk."""

    img = imread(path, mode='F')
    img /= np.max(img)
    return img
