from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from matplotlib.image import imread
# from sklearn.metrics import accuracy_score
# import matplotlib as plt
import numpy as np
import glob
import pickle
import sys

names = ["Nearest Neighbor", "Linear SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB()]


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
    flag = True
    while flag:
        print("--------------")
        a = input("Begin Training? y/n :")
        if a == 'y' or a == 'Y':
            train(t_images, t_labels, v_images, v_labels, te_images, te_labels)
            flag = False
        elif a == 'n' or a == 'N':
            flag = False
        else:
            print("Invalid input \n")

    flag = True
    while flag:
        print("--------------")
        a = input("Begin Prediction? y/n :")
        if a == 'y' or a == 'Y':
            predict(t_images, t_labels, v_images, v_labels, te_images, te_labels)
            flag = False
        elif a == 'n' or a == 'N':
            flag = False
        else:
            print("Invalid input \n")


    # flag = True
    # while flag:
    #     a = input("Predict (E)")
    #     if a == 'E':
    #         flag = False


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

    print("\n")

    for i in range(0, len(classifiers)):

        print("Training ", str(names[i]))
        current_classifier = classifiers[i]
        current_classifier.fit(t_images, t_labels)
        print("Saving model...\n")
        try:
            ext = ".pkl"
            pkl_filename = str(names[i]+ext)
            with open(pkl_filename, 'wb') as file:
                pickle.dump(current_classifier, file)
        except:
            print("Error", sys.exc_info()[0], "occurred.")

        i += 1


def predict(t_images, t_labels, v_images, v_labels, te_images, te_labels):
    ext = ".pkl"
    flag = True
    while flag:
        print("Prediction Main Menu")

        print("1) Nearest Neighbor")
        print("2) Linear SVM")
        print("3) Gaussian Process")
        print("4) Decision Tree")
        print("5) Random Forest")
        print("6) Neural Net")
        print("7) AdaBoost")
        print("8) Naive Bayes")
        a = input("Which Classifier would you like to use?")
        a = int(a)
        if a in range(1, 8):
            print("prediction using: ", names[a-1])
            # Load from file
            filename = names[a] + ext
            with open(filename, 'rb') as file:
                pickle_model = pickle.load(file)

            # Calculate the accuracy score and predict target values
            score = pickle_model.score(te_images, te_labels)
            print("Test score: ", (score*100))
            out = pickle_model.predict(te_images)

            for i in range(0, len(out)):
                print(out[i], te_labels[i])
        else:
            print("invalid input")


    # print("Training K-nearest neighbour classifier...")
    # knn_classifier = KNeighborsClassifier(3)
    # knn_classifier.fit(t_images, t_labels)
    # accuracy = knn_classifier.score(v_images, v_labels)
    # print("\nKKN Accuracy = " + str(accuracy * 100) + "%")
    # print("\nKNN Test Images Prediction = ", knn_classifier.predict(te_images), "Ground Truth = ", te_labels)
    #
    # print("Training random forest classifier:")
    # r_forest_classifier = RandomForestClassifier(3)
    # r_forest_classifier.fit(t_images, t_labels)
    # accuracy = r_forest_classifier.score(v_images, v_labels)
    # print("\nRandom Forest Accuracy = " + str(accuracy * 100) + "%\n")
    # print("Random Forest Test Images Prediction = ", r_forest_classifier.predict(te_images),
    #       "Ground Truth = ", te_labels)


