import pandas as pd
import numpy as np

import os
import time
import flickrapi
import urllib
import yaml

OUTPUT_FOLDER = './photos/'

def main():

    """
    Download the photos in the 'beauty' dataset.
    """

    # Load the API credentials
    with open('./flickr_api.txt') as f:
        keys = yaml.safe_load(f)

    # Set the API credentials
    flickr = flickrapi.FlickrAPI(keys['key'], keys['secret'])

    # Load the data
    df = pd.read_csv('./beauty-icwsm15-dataset.tsv', sep="\t", index_col=False)
    total_images = df.shape[0] * 1.0
    df['downloaded'] = None

    query_counter = 0.0
    for i, photo_id in enumerate(df['#flickr_photo_id']):
        if query_counter % 100.0 == 0:
            print(str(i) + '/' + str(total_images) + ' images (i.e. ' +
                  str(np.round(i / total_images, 3) * 100) + "%) complete.")
            time.sleep(15)
        path = OUTPUT_FOLDER + str(photo_id) + ".jpg"
        if os.path.exists(path):
            df.ix[i, 'downloaded'] = True
            continue
        try:
            query_counter += 1.0
            photo_response = flickr.photos.getInfo(photo_id=photo_id)
            download_photo(photo_id, photo_response)
            df.ix[i, 'downloaded'] = True
        except flickrapi.exceptions.FlickrError:
            df.ix[i, 'downloaded'] = False
            continue

    df.to_csv('./download_summary.tsv', sep="\t", index=False)


def download_photo(photo_id, photo_response):

    if len(photo_response) > 1:
        print("More than one photo available for ID: " + str(photo_id) + '. Choosing the first.')

    photo = photo_response[0]

    if photo.attrib.get('id') is None:
        print("Could not find photo ID. Skipping download of ID: " + str(photo_id))

    url = 'https://farm{farm_id}.staticflickr.com/{server_id}/{id}_{secret}.jpg'.format(
        farm_id=photo.attrib.get('farm'),
        server_id=photo.attrib.get('server'),
        id=photo.attrib.get('id'),
        secret=photo.attrib.get('secret')
    )

    try:
        urllib.request.urlretrieve(url, OUTPUT_FOLDER + photo.attrib.get('id') + ".jpg")
    except Exception as e:
        print(e, 'Download failure')


if __name__ == '__main__':
    main()
