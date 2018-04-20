"""
Useful functions for working with image data.
"""

import pandas as pd
import numpy as np
from scipy.misc import imread

def load_descriptions(base_path=None):

    """
    Load the image descriptions from the download summary.

    :param base_path: Path that contains the download summary and the photo directory
    :return: descriptions dataframe
    """
    if base_path is None: base_path = ''

    df = pd.read_csv(base_path + './download_summary.tsv', sep="\t", index_col=False)

    df = df[df.downloaded == True]

    df = df.rename(columns={'#flickr_photo_id': 'flickr_photo_id'})
    df['img_path'] = df.flickr_photo_id.apply(
        lambda x: base_path + './photos/' + str(x) + '.jpg'
    )
    return df


def read_colour_img(path):

    """
    Read colour image from disk.
    :param path: Path to image
    :return: (np.array) 2D array of image
    """

    return imread(path, mode='RGB')


def read_grayscale_img(path):

    """
    Read gray-scale image from disk.
    :param path: Path to image
    :return: (np.array) 2D array of image in gray-scale
    """

    img = imread(path, mode='F')
    img /= np.max(img)
    return img
