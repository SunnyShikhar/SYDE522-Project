import matplotlib.image as mpimg
import os
import matplotlib.pylab as plt
import numpy as np
import prepare_features
import pandas as pd
from scipy.stats import gaussian_kde
from scipy import stats

def get_image_sizes(base_path=None, output_path='./imgSize.tsv'):
    df = prepare_features.load_descriptions(base_path)
    df['img_height'] = None
    df['img_width'] = None
    df['img_area'] = None

    total_images = df.shape[0]
    print(total_images)
    images_processed = 0

    for idx, row in df.iterrows():
        # Provide a status update if necessary
        if images_processed % 100.0 == 0:
            print(str(images_processed) + '/' + str(total_images) + ' images (i.e. ' +
                  str(np.round(images_processed / total_images, 3) * 100) + "%) complete.")

        # Extract LBP features for image
        img = mpimg.imread(row.img_path)
        # print(img.shape)

        df.at[idx, 'img_height'] = img.shape[0]
        df.at[idx, 'img_width'] = img.shape[1]
        df.at[idx, 'img_area'] = img.shape[0]*img.shape[1]

        images_processed += 1

    df['avg_beauty_score'] = df.beauty_scores.apply(lambda x: np.mean(np.asarray([int(i) for i in x.split(',')])))
    df['std_beauty_score'] = df.beauty_scores.apply(lambda x: np.std(np.asarray([int(i) for i in x.split(',')])))
    
    # Write the dataframe to disk
    if output_path is not None:
        df.to_csv(output_path, index=False, sep="\t")

    return 

#This is a function that will plot a histogram based on the data field provided
def plot_histogram(dataOnX, XName, norm=True):
    weights = np.ones_like(dataOnX)/float(len(dataOnX))
    plt.hist(dataOnX, weights = weights)
    plt.xlabel(XName)
    plt.title(XName + ' Histogram')

    if norm == True:
        # Fit a normal distribution to the data:
        mu, std = stats.norm.fit(dataOnX)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, len(dataOnX))
        p = stats.norm.pdf(x, mu, std)
        plt.plot(x, p, 'k')
        title = XName + 'Histogram' + ", mean = %.2f,  standard deviation = %.2f" % (mu, std)
        plt.title(title)

    plt.savefig('./plots/' + XName + '.png')
    plt.show()

def scatter_plot(x, y, title, yaxis, xaxis):
    plt.scatter(x, y, color= 'red', alpha=0.3)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.axis()
    plt.legend()
    plt.show()


def plot_image_size_hist(imgDataframePath='./imgSize.tsv'):
    imgDF = pd.read_csv(imgDataframePath, sep='\t')

    # Plot height distribution
    plot_histogram(imgDF['img_height'], 'Image Height')

    plot_histogram(imgDF['img_width'], 'Image Width')

    plot_histogram(imgDF['img_area'], 'Image Area')

    plot_histogram(imgDF['avg_beauty_score'], 'Standard Deviation of Beauty Score')
    plot_histogram(imgDF[imgDF.category == 'urban']['avg_beauty_score'], 'Average Beauty Score for Urban')
    plot_histogram(imgDF[imgDF.category == 'nature']['avg_beauty_score'], 'Average Beauty Score for Nature')
    plot_histogram(imgDF[imgDF.category == 'people']['avg_beauty_score'], 'Average Beauty Score for People')
    plot_histogram(imgDF[imgDF.category == 'animals']['avg_beauty_score'], 'Average Beauty Score for Animals')

    plot_histogram(imgDF['std_beauty_score'], 'Standard Deviation of Beauty Score', False)
    plot_histogram(imgDF[imgDF.category == 'urban']['std_beauty_score'], 'Standard Deviation of Beauty Score for Urban', False)
    plot_histogram(imgDF[imgDF.category == 'nature']['std_beauty_score'], 'Standard Deviation of Beauty Score for Nature', False)
    plot_histogram(imgDF[imgDF.category == 'people']['std_beauty_score'], 'Standard Deviation of Beauty Score for People', False)
    plot_histogram(imgDF[imgDF.category == 'animals']['std_beauty_score'], 'Standard Deviation of Beauty Score for Animals', False)

    # scatter_plot(imgDF['img_width'], imgDF['img_height'], 'Image height vs width', 'height', 'width')

    # Calculate point density
    densityPlot = np.vstack([imgDF['img_width'], imgDF['img_height']])
    z = gaussian_kde(densityPlot)(densityPlot)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = imgDF['img_width'][idx], imgDF['img_height'][idx], z[idx]

    # Plot Scatter plot based on density
    fig, ax = plt.subplots()
    sc = ax.scatter(x, y, c=z, s=50, edgecolor='', alpha=0.3, cmap=plt.cm.jet)
    plt.title('Image height vs width')
    plt.xlabel('width')
    plt.ylabel('height')
    plt.axis()
    plt.legend()
    plt.colorbar(sc)
    plt.show()



if __name__ == '__main__':

    # Get image size for all the pictures and output to csv
    # image_df = get_image_sizes()

    # Plot graph 
    plot_image_size_hist()