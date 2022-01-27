import numpy as np
import pandas as pd
from scipy.special import softmax as sm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# np.random.seed(100)
# test = pd.read_csv("train.csv")
# test = test.sort_values(by="LABELS")
# test = test.sort_values(by="LABELS",ascending=False)
# test = test.drop(["time","lat","lon"],axis=1)
# test = test.sample(frac=1)
# test = test.drop(["S.No"],axis=1)
# test_array = np.array(test)
#
#
# train = pd.read_csv("test.csv")
# sno = np.array(np.array(train)[:,0])
# train = train.drop(["S.No"],axis=1)
# train = train.drop(["time","lat","lon"],axis=1)
# train_array = np.array(train)

class SoftmaxRegression:

    def __init__(self, y_labels, x_train):
        self.y_labels = y_labels
        self.x_train = x_train
        self.nb_classe = len(np.unique(y_labels))

    # Cette fonction va trouver le w et le b qui sont optimals.
    def train(self, eta, lam, nb_iterations):

        w_current = np.random.random([len(x_train[1, :]), 3])
        b_current = np.random.random(1)
        y_onehot = onehot(self.y_labels, self.nb_classe)

        for i in range(nb_iterations):
            z = np.dot(x_train, w_current) + b_current
            y_hat = np.array([softmax_vect(x) for x in z])
            w_grad = (1 / len(self.x_train[:, 0])) * np.dot(np.transpose(self.x_train), (y_hat - y_onehot)) + 2 * lam * w_current
            b_grad = (1 / len(self.x_train[:, 0])) * np.sum(y_hat - y_onehot)
            w_current = w_current - eta * w_grad
            b_current = b_current - eta * b_grad
            self.w = w_current
            self.b = b_current
        self.w = w_current
        self.b = b_current

    # Fonction qui retourne un vecteur contenant les classes de toutes les prédictions.
    def predict(self, x_test):
        z = np.dot(x_test, self.w) + self.b
        y_onehot = onehot(self.y_labels, self.nb_classe)
        y_hat = np.array([softmax_vect(x) for x in z])
        class_pred = [np.argmax(y_hat[i]) for i in range(len(y_hat))]
        return class_pred

# Permet l'encodage onehot de n'importe quel scalaire n_class est simplement le nombre de classe qui va
# donc nous donner la longueur que le vecteur final aura.
def onehot(x,n_class):
    encoded = []
    for i in x:
        yi_encoded = np.zeros(n_class)
        yi_encoded[int(i)] = 1
        encoded.append(yi_encoded)
    return np.array(encoded)

# Prends un vecteur et retourne un vecteur où a été appliqué la fonction softmax sur tous les éléments du vecteur x.
def softmax_vect(x):
    y = np.exp(x-np.max(x))
    div = np.sum(y)
    return y/div

# Retourne la proportion de bonne prédictions trouver avec un certain estimateur. y_pred et y_vrai ne doivent pas être
# encodé en onehot.
def accuracy(y_pred, y_vrai):
    somme = 0
    for i in range(len(y_vrai)):
        if np.array_equal(y_vrai[i],y_pred[i]):
            somme+=1
    return somme / len(y_vrai)

#
# var_explicative = test_array[:,0:len(test_array[0])-1]
#
# prop_train = np.floor(len(var_explicative[:, 0])*.80)
# prop_validation =  len(var_explicative[:, 0]) - prop_train
#
# labels = test_array[:,-1]
#
# x_train = var_explicative[0:int(prop_train)]
# x_validation = var_explicative[int(prop_train):len(var_explicative[:, 0])]
#
# labels_train = labels[0:int(prop_train)]
# labels_validation = labels[int(prop_train):len(var_explicative[:, 0])]
#
# labels_train_soh = labels_train
# labels_validation_soh = labels_validation
#
# labels_train = onehot(labels_train,3)
# labels_validation = onehot(labels_validation,3)
#
# nb_iteration = 11
# regression_softmax = SoftmaxRegression(labels_train_soh, x_train)
# regression_softmax.train(0.000001, 0.0001, nb_iteration)
# class_pred_validation = regression_softmax.predict(x_validation)
# print(np.unique(class_pred_validation, return_counts=True))
#
# print(accuracy(regression_softmax.predict(x_validation), labels_validation_soh))

#
# knn = KNeighborsClassifier(n_neighbors=5)
# knn_3 = KNeighborsClassifier(n_neighbors=3)
# knn_7 = KNeighborsClassifier(n_neighbors=7)
# knn_9 = KNeighborsClassifier(n_neighbors=9)
# knn_11 = KNeighborsClassifier(n_neighbors=11)
# knn_13 = KNeighborsClassifier(n_neighbors=13)
# knn_15 = KNeighborsClassifier(n_neighbors=15)
# knn.fit(x_train, labels_train_soh)
# knn_3.fit(x_train, labels_train_soh)
# knn_7.fit(x_train, labels_train_soh)
# knn_9.fit(x_train, labels_train_soh)
# knn_11.fit(x_train, labels_train_soh)
# knn_13.fit(x_train, labels_train_soh)
# knn_15.fit(x_train, labels_train_soh)
# predictions_knn = [knn.predict([x]) for x in x_validation]
# predictions_knn_3 = [knn_3.predict([x]) for x in x_validation]
# predictions_knn_7 = [knn_7.predict([x]) for x in x_validation]
# predictions_knn_9 = [knn_9.predict([x]) for x in x_validation]
# predictions_knn_11 = [knn_11.predict([x]) for x in x_validation]
# predictions_knn_13 = [knn_13.predict([x]) for x in x_validation]
# predictions_knn_15 = [knn_15.predict([x]) for x in x_validation]
#
# predictions_knn = [predictions_knn[i][0] for i in range(len(predictions_knn))]
# predictions_knn_3 = [predictions_knn_3[i][0] for i in range(len(predictions_knn))]
# predictions_knn_7 = [predictions_knn_7[i][0] for i in range(len(predictions_knn))]
# predictions_knn_9 = [predictions_knn_9[i][0] for i in range(len(predictions_knn))]
# predictions_knn_11 = [predictions_knn_11[i][0] for i in range(len(predictions_knn))]
# predictions_knn_13 = [predictions_knn_13[i][0] for i in range(len(predictions_knn))]
# predictions_knn_15 = [predictions_knn_15[i][0] for i in range(len(predictions_knn))]
#
# acc_knn_all = [accuracy(predictions_knn_3,labels_validation_soh),accuracy(predictions_knn,labels_validation_soh),accuracy(predictions_knn_7,labels_validation_soh),
#                accuracy(predictions_knn_9,labels_validation_soh),accuracy(predictions_knn_11,labels_validation_soh),accuracy(predictions_knn_13,labels_validation_soh),
#                accuracy(predictions_knn_15,labels_validation_soh)]
# k = [3,5,7,9,11,13,15]
# print(acc_knn_all)
# print(np.unique(predictions_knn_7,return_counts=True))
# print(np.unique(predictions_knn_15,return_counts=True))
# plt.plot(k,acc_knn_all)
# plt.show()
#
# print("knn", np.unique(predictions_knn,return_counts=True))
# predictions_knn = [predictions_knn[i][0] for i in range(len(predictions_knn))]
# predictions_knn_test = [knn.predict([x]) for x in train_array]
# predictions_knn_test = [predictions_knn_test[i][0] for i in range(len(predictions_knn_test))]
# print(predictions_knn)
# print(np.unique(predictions_knn_test,return_counts=True))
# print(accuracy(predictions_knn, labels_validation_soh))

