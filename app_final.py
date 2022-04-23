import streamlit as st
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score,GridSearchCV
import pandas as pd
import argparse
import cv2
import numpy as np
import matplotlib.image as pltimg
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score,GridSearchCV, StratifiedShuffleSplit, RandomizedSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
from sklearn.svm import SVC
import tensorflow as tf
from PIL import Image
import colorsys


df = pd.read_csv('dataset_esst.csv', dtype=int)
st.image('https://eestechchallenge.eestec.pp.ua/assets/images/ec-logo.png')
st.sidebar.header('EESTech Challenge Porto 2022')
def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


color_chosen=st.sidebar.color_picker(label='Pick a color')


img='data:image/png;base64,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'
#st.sidebar.image(logo)
cod_rgbs=np.asarray(hex_to_rgb(color_chosen))
sat=st.sidebar.slider('Saturação',0.0,1.0,step=0.1)














if color_chosen=='#000000':
    st.image('unknown.jpg')
    k = 5
    # load the image and convert it from BGR to RGB so that
    # we can dispaly it with matplotlib
    image = cv2.imread("unknown.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # show our image
    plt.figure()
    plt.axis("off")
    plt.imshow(image)
    #plt.show()

    # reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # cluster the pixel intensities
    clt = KMeans(n_clusters=k)
    clt.fit(image)


    def centroid_histogram(clt):
        # grab the number of different clusters and create a histogram
        # based on the number of pixels assigned to each cluster
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins=numLabels)
        # normalize the histogram, such that it sums to one
        hist = hist.astype("float")
        hist /= hist.sum()
        # return the histogram
        return hist


    def plot_colors(hist, centroids):
        # initialize the bar chart representing the relative frequency
        # of each of the colors
        bar = np.zeros((50, 300, 3), dtype="uint8")
        startX = 0
        # loop over the percentage of each cluster and the color of
        # each cluster
        for (percent, color) in zip(hist, centroids):
            # plot the relative percentage of each cluster
            endX = startX + (percent * 300)
            cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                          color.astype("uint8").tolist(), -1)
            startX = endX

        # return the bar chart
        return bar


    # build a histogram of clusters and then create a figure
    # representing the number of pixels labeled to each color
    hist = centroid_histogram(clt)
    bar = plot_colors(hist, clt.cluster_centers_)
    # show our color bart
    plt.figure()
    plt.axis("off")
    plt.imshow(bar)
    plt.imshow(bar)
    plt.savefig('my_plot.png')
    d1, d2, d3= st.columns([1,10,1])
    with d1:
       st.write('')
    with d2:
        st.image('my_plot.png')
    with d3:
       st.write('')
else:
    knn = KNeighborsClassifier()
    param_grid = {'n_neighbors': [3, 5, 7, 9],
                  'weights': ['uniform', 'distance'],
                  'algorithm': ['auto', 'ball_tree', 'kd_tree'],
                  'p': [1, 2]}
    clf_knn = GridSearchCV(knn, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
    y = np.arange(0, 510, 1)
    print(y)
    knn.fit(df, y)
    index = knn.predict([cod_rgbs])[0]
    palets = {}
    palettes = np.array(df).reshape((int(510 / 5), 5, -1))
    for i in range(len(palettes)):
        palets.update({str(i): palettes[i]})
    index = knn.predict([cod_rgbs])[0]
    p = str(int(index / 5))
    pal=palets[p]
    c1 = colorsys.rgb_to_hsv(pal[0][0], pal[0][1],
                             pal[0][2])
    c2 = colorsys.rgb_to_hsv(pal[1][0], pal[1][1],
                             pal[1][2])
    c3 = colorsys.rgb_to_hsv(pal[2][0], pal[2][1],
                             pal[2][2])
    c4 = colorsys.rgb_to_hsv(pal[3][0], pal[3][1],
                             pal[3][2])
    c5 = colorsys.rgb_to_hsv(pal[4][0], pal[4][1],
                             pal[4][2])

    img = Image.open("unknown.jpg")
    img = img.convert("HSV")
    p1 = palets[p]
    d = img.getdata()

    new_image = []





    for item in d:
        # change all white (also shades of whites)
        # pixels to yellow

        newitem = colorsys.rgb_to_hsv(item[0] / 255, item[1] / 255, item[2] / 255)

        if c1[0] - 0.05 < newitem[0] < c1[0] + 0.05:
            new_image.append(tuple(p1[0]))
        elif c2[0] - 0.05 < newitem[0] < c2[0] + 0.05:
            new_image.append(tuple(p1[1]))
        elif c3[0] - 0.05 < newitem[0] < c3[0] + 0.05:
            new_image.append(tuple(p1[2]))
        elif c4[0] - 0.05 < newitem[0] < c4[0] + 0.05:
            new_image.append(tuple(p1[3]))
        elif c5[0] - 0.05 < newitem[0] < c5[0] + 0.05:
            new_image.append(tuple(p1[4]))
        elif 170 / 360 < newitem[0] < 180 / 360:
            new_image.append(tuple(p1[2]))
        else:
            new_image.append(item)


    # update image data
    img.putdata(new_image)
    img = img.convert("RGB")
    # save new image
    img.save("imgaltered.jpg")
    st.image("imgaltered.jpg")

    k = 5
    # load the image and convert it from BGR to RGB so that
    # we can dispaly it with matplotlib
    image = cv2.imread("imgaltered.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # show our image
    plt.figure()
    plt.axis("off")
    plt.imshow(image)
    # plt.show()

    # reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # cluster the pixel intensities
    clt = KMeans(n_clusters=k)
    clt.fit(image)


    def centroid_histogram(clt):
        # grab the number of different clusters and create a histogram
        # based on the number of pixels assigned to each cluster
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins=numLabels)
        # normalize the histogram, such that it sums to one
        hist = hist.astype("float")
        hist /= hist.sum()
        # return the histogram
        return hist


    def plot_colors(hist, centroids):
        # initialize the bar chart representing the relative frequency
        # of each of the colors
        bar = np.zeros((50, 300, 3), dtype="uint8")
        startX = 0
        # loop over the percentage of each cluster and the color of
        # each cluster
        for (percent, color) in zip(hist, centroids):
            # plot the relative percentage of each cluster
            endX = startX + (percent * 300)
            cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                          color.astype("uint8").tolist(), -1)
            startX = endX

        # return the bar chart
        return bar


    # build a histogram of clusters and then create a figure
    # representing the number of pixels labeled to each color
    hist = np.array([1/5,1/5,1/5,1/5,1/5])
    bar = plot_colors(hist, clt.cluster_centers_)
    bar_p = plot_colors(hist, pal)
    plt.imshow(bar_p)
    plt.savefig('palete_input.jpg')
    st.sidebar.image('palete_input.jpg')
    logo = st.sidebar.file_uploader('Upload a logo')
    # show our color bart
    plt.figure()
    plt.axis("off")
    plt.imshow(bar)
    plt.imshow(bar)
    plt.savefig('my_plot.png')
    d1, d2, d3 = st.columns([1, 10, 1])
    with d1:
        st.write('')
    with d2:
        st.image('my_plot.png')
    with d3:
        st.write('')


