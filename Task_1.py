from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from matplotlib.image import imread
import glob


def task1():
    print('\nTASK 1')
    print('------')
    tr_dir = 'digits-train-5000'
    te_dir = 'digits-test'
    v_dir = 'digits-validation-1000'
    f_ext = '/*.png'

    t_images, t_labels = load_dataset(tr_dir, f_ext)
    te_images, te_labels = load_dataset(te_dir, f_ext)
    v_images, v_labels = load_dataset(v_dir, f_ext)
    t_images = reshape(t_images)
    te_images = reshape(te_images)
    v_images = reshape(v_images)
    train(t_images, t_labels, v_images, v_labels, te_images, te_labels)


def load_dataset(set_n, ext):
    data_set = set_n + ext

    labels = []
    images = []

    print("loading dataset " + set_n + "...")

    for file in glob.glob(data_set):
        images.append(imread(file))
        label = len(set_n)
        label += 1
        labels.append(file[label])
    return images, labels


def reshape(arr):
    arr = np.array(arr)
    # print(arr.shape)
    n_samples, nx, ny = arr.shape
    d2_arr = arr.reshape((n_samples, nx * ny))
    # print(d2_arr.shape)
    return d2_arr


def train(t_images, t_labels, v_images, v_labels, te_images, te_labels):

    print("Training K-nearest neighbour classifier...")
    knn_classifier = KNeighborsClassifier(3)
    knn_classifier.fit(t_images, t_labels)
    accuracy = knn_classifier.score(v_images, v_labels)
    print("\nKKN Accuracy = " + str(accuracy * 100) + "%")
    print("\nKNN Test Prediction = ", knn_classifier.predict(te_images), "Ground Truth = ", te_labels)



    # print("Training random forest classifier:")
    # r_forest_classifier = RandomForestClassifier(3)
    # r_forest_classifier.fit(t_images, t_labels)
    # accuracy = r_forest_classifier.score(v_images, v_labels)
    # print("\nRandom Forest Accuracy = " + str(accuracy * 100) + "%\n")
    # # print(r_forest_classifier.predict(testimages))

