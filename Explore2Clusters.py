import os
import sklearn.cluster as cluster
import logging
from time import time
import matplotlib.pyplot as plt
from sklearn import decomposition


def MainInforExtract(dataset, image_shape):
    """
    :param dataset: A numpy array. (n_samples, n_features), the input images data must gray scale.
    :param image_shape: the input images shape
    :return: No return
    """
    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    n_row, n_col = 1, 1
    n_components = n_row * n_col

    # #############################################################################
    # Load faces data
    n_samples, n_features = dataset.shape

    # global centering
    data_centered = dataset - dataset.mean(axis=0)

    # local centering
    data_centered = data_centered - data_centered.mean(axis=1).reshape(n_samples, -1)

    print("Dataset consists of %d samples" % n_samples)

    def plot_gallery(title, images, n_col = n_col, n_row = n_row):
        plt.figure(figsize=(2. * n_col, 2.26 * n_row))
        plt.suptitle(title, size=16)
        for i, comp in enumerate(images):
            maxV = np.max(comp)
            minV = np.min(comp)
            comp = (comp - minV) / (maxV - minV)
            plt.subplot(n_row, n_col, i + 1)
            plt.imshow(comp.reshape(image_shape))
            plt.xticks(())
            plt.yticks(())
        plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)


    # #############################################################################
    # List of the different estimators, whether to center and transpose the
    # problem, and whether the transformer uses the clustering API.
    estimators = [

        ('MiniBatchDictionaryLearning',
         decomposition.MiniBatchDictionaryLearning(n_components=12, alpha=0.12, random_state=512, batch_size=4, n_iter=3090),
         True),
    ]


    # #############################################################################
    # Do the estimation and plot it

    for name, estimator, center in estimators:
        print("Extracting the top %d %s..." % (n_components, name))
        t0 = time()
        data = dataset
        if center:
            data = data_centered
        print("Input data shape is {}".format(data.shape))
        estimator.fit(data)
        train_time = (time() - t0)
        print("done in %0.3fs" % train_time)
        if hasattr(estimator, 'cluster_centers_'):
            components_ = estimator.cluster_centers_
        else:
            components_ = estimator.components_
        print("Components shape {}".format(components_.shape))

        # Plot an image representing the pixelwise variance provided by the
        # estimator e.g its noise_variance_ attribute. The Eigenfaces estimator,
        # via the PCA decomposition, also provides a scalar noise_variance_
        # (the mean of pixelwise variance) that cannot be displayed as an image
        # so we skip it.

        plot_gallery('%s - Train time %.1fs' % (name, train_time),
                     components_[:n_components])

    plt.show()

if __name__ == "__main__":
    import os
    import numpy as np
    imagePath = "CropResizeTest"
    imgNames = sorted(list(os.listdir(imagePath)))
    imgNames = np.array(imgNames)
    tsneOriImage = np.load("./OriTsneResult.npy")

    ### select perspective
    perspective = [1,3]
    tsneOriSelectPerspective = []
    tsneOriSelectName = []
    print(len(imgNames))
    print(tsneOriImage.shape)
    for i,name in enumerate(imgNames):
        pers = int(name.split(".jpg")[0][-1])
        if pers in perspective and i != (len(imgNames)-1):
            tsneOriSelectPerspective.append(tsneOriImage[i])
            tsneOriSelectName.append(name)

    tsneOriSelectPerspective = np.array(tsneOriSelectPerspective)
    print("Selected Perspective number is {}".format(tsneOriSelectPerspective.shape))
    dbscan = cluster.DBSCAN(eps=0.25, min_samples=3, leaf_size=50)
    dbscan = dbscan.fit(tsneOriSelectPerspective)
    labels = dbscan.labels_
    print("There are {} clusters.".format(np.unique(labels)))

    ### find the biggest and the second biggest clusters.
    label2num = {}
    for label in labels:
        if label not in label2num:
            label2num[label] = 1
        else:
            num = label2num[label]
            num += 1
            label2num[label] = num
    keys = []
    values = []
    for key, value in label2num.items():
        keys.append(key)
        values.append(value)
    maxIndex = np.argmax(values)
    values.pop(int(maxIndex))
    secondMaxIndex = np.argmax(values)

    largestLabel = keys[int(maxIndex)]
    secondMaxLabel = keys[int(secondMaxIndex)]

    largestIndices = np.where(labels == largestLabel)[0]
    secondMaxIndices = np.where(labels == secondMaxLabel)[0]

    drawPlot = np.concatenate([tsneOriSelectPerspective[largestIndices],
                               tsneOriSelectPerspective[secondMaxIndices]], axis=0)
    drawPlotLabels = np.concatenate([labels[largestIndices],
                                     labels[secondMaxIndices]], axis=0)

    ## Draw DBSCAN labeled
    _, ax = plt.subplots()
    ax.scatter(tsneOriSelectPerspective[:,0], tsneOriSelectPerspective[:,1], c = labels)
    plt.show()


    ## Draw DBSCAN labeled cluster
    fig, axO = plt.subplots()
    oriX = drawPlot[:,0]
    oriY = drawPlot[:,1]
    axO.scatter(oriX, oriY,c = drawPlotLabels)
    axO.set_title("DBSCAN t-SNE clusters")
    plt.show()

    ## select some plots to show
    random = np.random.randint(0,len(largestIndices),size=[10])
    from PIL import Image
    f, axarr = plt.subplots(5,2)
    axarr[0,0].imshow(Image.open(os.path.join(imagePath, tsneOriSelectName[largestIndices[random[0]]])))
    axarr[0,1].imshow(Image.open(os.path.join(imagePath, tsneOriSelectName[largestIndices[random[1]]])))
    axarr[1,0].imshow(Image.open(os.path.join(imagePath, tsneOriSelectName[largestIndices[random[2]]])))
    axarr[1,1].imshow(Image.open(os.path.join(imagePath, tsneOriSelectName[largestIndices[random[3]]])))
    axarr[2,0].imshow(Image.open(os.path.join(imagePath, tsneOriSelectName[largestIndices[random[4]]])))
    axarr[2,1].imshow(Image.open(os.path.join(imagePath, tsneOriSelectName[largestIndices[random[5]]])))
    axarr[3,0].imshow(Image.open(os.path.join(imagePath, tsneOriSelectName[largestIndices[random[6]]])))
    axarr[3,1].imshow(Image.open(os.path.join(imagePath, tsneOriSelectName[largestIndices[random[7]]])))
    axarr[4,0].imshow(Image.open(os.path.join(imagePath, tsneOriSelectName[largestIndices[random[8]]])))
    axarr[4,1].imshow(Image.open(os.path.join(imagePath, tsneOriSelectName[largestIndices[random[9]]])))
    f.suptitle("Label 1")
    plt.show()

    random = np.random.randint(0,len(secondMaxIndices), size=[10])
    from PIL import Image, ImageOps
    f1, axarr = plt.subplots(5,2)
    axarr[0,0].imshow(Image.open(os.path.join(imagePath, tsneOriSelectName[secondMaxIndices[random[0]]])))
    axarr[0,1].imshow(Image.open(os.path.join(imagePath, tsneOriSelectName[secondMaxIndices[random[1]]])))
    axarr[1,0].imshow(Image.open(os.path.join(imagePath, tsneOriSelectName[secondMaxIndices[random[2]]])))
    axarr[1,1].imshow(Image.open(os.path.join(imagePath, tsneOriSelectName[secondMaxIndices[random[3]]])))
    axarr[2,0].imshow(Image.open(os.path.join(imagePath, tsneOriSelectName[secondMaxIndices[random[4]]])))
    axarr[2,1].imshow(Image.open(os.path.join(imagePath, tsneOriSelectName[secondMaxIndices[random[5]]])))
    axarr[3,0].imshow(Image.open(os.path.join(imagePath, tsneOriSelectName[secondMaxIndices[random[6]]])))
    axarr[3,1].imshow(Image.open(os.path.join(imagePath, tsneOriSelectName[secondMaxIndices[random[7]]])))
    axarr[4,0].imshow(Image.open(os.path.join(imagePath, tsneOriSelectName[secondMaxIndices[random[8]]])))
    axarr[4,1].imshow(Image.open(os.path.join(imagePath, tsneOriSelectName[secondMaxIndices[random[9]]])))
    f1.suptitle("Label 2")
    plt.show()

    ### Draw main information
    #### one label main infor
    label1Imgs = []
    label1OriImgs = []
    for index in largestIndices:
        label1Imgs.append(np.array(ImageOps.grayscale(Image.open(os.path.join(imagePath, tsneOriSelectName[index])))).reshape([-1]))
        label1OriImgs.append(np.array(Image.open(os.path.join(imagePath, tsneOriSelectName[index])).convert("RGB")))
    label1Imgs = np.array(label1Imgs)
    label1OriImgs = np.array(label1OriImgs) / 255.
    print(label1Imgs.shape)
    print(label1OriImgs[0,0,0])
    _, mean1 = plt.subplots()
    mean1.imshow(np.mean(label1OriImgs, axis=0, keepdims=False))
    plt.show()
    #MainInforExtract(label1Imgs,[96, 416])


    ### two label main infor
    label2Imgs = []
    label2OriImgs = []
    for index in secondMaxIndices:
        label2Imgs.append(np.array(ImageOps.grayscale(Image.open(os.path.join(imagePath, tsneOriSelectName[index])))).reshape([-1]))
        label2OriImgs.append(np.array(Image.open(os.path.join(imagePath, tsneOriSelectName[index])).convert("RGB")))
    label2Imgs = np.array(label2Imgs)
    label2OriImgs = np.array(label2OriImgs) / 255.
    print(label2Imgs.shape)
    print(label2OriImgs[0,0,0])
    _, mean2 = plt.subplots()
    mean2.imshow(np.mean(label2OriImgs, axis=0, keepdims=False))
    plt.show()
    #MainInforExtract(label2Imgs,[96, 416])

