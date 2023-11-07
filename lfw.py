from sklearn.datasets import fetch_lfw_people
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
from l12dpcanet import *
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV



#read in images and labels
faces = fetch_lfw_people(min_faces_per_person=10)
X = faces.images
Y = faces.target
labels = np.unique(Y)

N = X.shape[0]
X_new = np.zeros((N,64,47))

for i in range(N):
    img = X[i]
    img = cv2.resize(img,(47,64))
    X_new[i] = img


tsize = 7
x_train = []
y_train = []
x_test = []
y_test = []
random.seed(1234)
for i in labels:
    idxs = list(np.where(Y==i)[0])
    #randomly partition the indexs into training and testing
    idxs_train = random.sample(idxs, tsize)
    idxs_test = [x for x in idxs if x not in idxs_train]
    x_train.append(X_new[idxs_train])
    x_test.append(X_new[idxs_test])
    y_train.append(Y[idxs_train])
    y_test.append(Y[idxs_test])
x_train = np.concatenate(x_train,axis=0)
x_test = np.concatenate(x_test,axis=0)
y_train = np.concatenate(y_train,axis=0)
y_test = np.concatenate(y_test,axis=0)

# apply l12dpcanet_features
filters = get_l12dpcanet_filters(x_train)
x_train_pca = get_l12dpcanet_features(x_train,filters)
x_test_pca = get_l12dpcanet_features(x_test,filters)

#x_train_pca = get_l12dpcanet_features_oldversion(x_train)
#x_test_pca = get_l12dpcanet_features_oldversion(x_test)



svc = svm.SVC()
parameters = [
    {
        'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
        'gamma': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],
        'kernel': ['rbf']
    },
    {
        'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
        'kernel': ['linear']
    }
]
clf = GridSearchCV(svc, parameters, cv=3)
clf.fit(x_train_pca, y_train)
print(clf.best_params_)
best_model = clf.best_estimator_
x_all_pca = np.concatenate((x_train_pca,x_test_pca),axis=0)
y_all = np.concatenate((y_train,y_test),axis=0)

y_pred  = best_model.predict(x_all_pca)

np.sum(y_all == y_pred)/len(y_all)

#Create a svm Classifier
#clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
#clf.fit(x_train_pca, y_train)

#Predict the response for test dataset
#y_pred = clf.predict(x_test_pca)
#np.sum(y_test == y_pred)/len(y_test)


'''
_, axes = plt.subplots(5, 3, figsize=(10, 6))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(x_train[i], cmap='gray')

    #ax.set_xlabel(f'label:{X_train[i]}')
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
'''

'''
# apply sklearn pca
n_samples = x_train.shape[0]
image_size = x_train.shape[1] * x_train.shape[2]
X_train = x_train.reshape(n_samples, image_size)
n_samples_test = x_test.shape[0]
image_size_test = x_test.shape[1] * x_test.shape[2]
X_test = x_test.reshape(n_samples_test, image_size_test)
pca=PCA(20).fit(X_train)
x_train_pca= pca.transform(X_train)
x_test_pca= pca.transform(X_test)
'''



'''
#PCA Net
from PCANet import pcanet as net

pcanet = net.PCANet(
    image_shape=(64,47),
    filter_shape_l1=2, step_shape_l1=1, n_l1_output=2,  # parameters for the 1st layer
    filter_shape_l2=2, step_shape_l2=1, n_l2_output=2,  # parameters for the 2nd layer
    filter_shape_pooling=2, step_shape_pooling=2        # parameters for the pooling layer
)
#pcanet.validate_structure()
pcanet.fit(x_train)  # Train PCANet

x_train_pca = pcanet.transform(x_train)
x_test_pca = pcanet.transform(x_test)


#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(x_train_pca, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test_pca)
np.sum(y_test == y_pred)/len(y_test)
'''












