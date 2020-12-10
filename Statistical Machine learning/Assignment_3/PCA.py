import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def PCA (train_data, dimension,test_data = []):
    mean_vector = np.mean(train_data,axis=0)
    mean_vector = mean_vector.reshape(len(mean_vector),1)

    centered_data = train_data - mean_vector.T

    covariance_matrix = np.cov(centered_data.T)
    eigen_value, eigen_vector = np.linalg.eig(covariance_matrix)

    for i in range(len(eigen_value)):
        eigv = eigen_vector[:,i].reshape(len(eigen_value),1)
        np.testing.assert_array_almost_equal(covariance_matrix.dot(eigv), eigen_value[i] * eigv,
                                            decimal=6, err_msg='', verbose=True)

                                            
    for ev in eigen_vector.T:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

    eigen_pair = [(eigen_value[i],eigen_vector[:,i]) for i in range(len(eigen_value))]

    eigen_pair.sort(key=lambda x: x[0], reverse=True)

    stack_vector = [eigen_vector[:,i] for i in range(dimension)]

    matrix_w = np.vstack(stack_vector)

    transformed_data = np.dot(train_data,matrix_w.T)
    if test_data == []:
        pass
    else:
        transformed_test_data = np.dot(test_data,matrix_w.T)
    if dimension == 784:
        return train_data, test_data
    return transformed_data.astype(float),transformed_test_data.astype(float)

def distance(pointA, pointB, _norm=np.linalg.norm):
    return _norm(pointA - pointB)

class K_mean:
    def __init__(self, k = 10):
        self.k = k
        self.centroids = None
        self.data = None
        self.label = None
        self.clusters = None
        self.cluster_acc = None
        self.loss = None

    def centroid_random_init(self):
        length_data = len(self.data)
        np.random.seed(3)
        random_index = np.random.choice(length_data, size = self.k,replace = False)
        #print(random_index)
        #print(random_index.shape)
        self.centroids = self.data[random_index,:]
        #print(self.centroids.shape)
    def centroid_init(self):
        length_data = len(self.data)
        np.random.seed(3)
        random_index_list = []
        for i in range(self.k):
            random_index = np.random.choice(length_data, size = 1,replace = False)
            while self.label[random_index] != i:
                random_index = np.random.choice(length_data, size = 1,replace = False)
            #print(self.label[random_index])
            random_index_list.append(random_index)
        self.centroids = self.data[random_index_list,:]
        #print(self.centroids.shape)
        
        
    def update_centroid(self):

        for i in range(self.k):
            
            index = np.where(self.clusters == i)[0]
            #print(index)
            #print(index.shape)

            #print(self.data[index,:].shape)
            #print(self.data[index,:].mean(axis=0).shape)
            self.centroids[i] = self.data[index,:].mean(axis=0)
            #self.centroids = np.mean(self.clusters)
        
    def fit(self, train_data ,label, n_iteration, random_init = True):
        
        self.data = train_data
        self.label = label
        n_data = self.data.shape[0]
        if random_init == True:
            self.centroid_random_init()
        else:
            self.centroid_init()
        Distance_to_centroids = 0
        self.clusters = np.zeros(shape = (n_data,1))
        self.loss = np.zeros(shape = (n_iteration,1))

        for iters in range(n_iteration):
            
            for i in range(n_data):
                min_distance = 10000000
                cluster_index = 0
                for j in range(self.k):    
                    Distance_to_centroids = distance(self.data[i,:],self.centroids[j])
                    #if Distance_to_centroids == 0:
                        #print(i)
                        #print(Distance_to_centroids)
                    if Distance_to_centroids < min_distance:
                        min_distance = Distance_to_centroids
                        cluster_index = j
                        #print
                self.clusters[i] = cluster_index
                self.loss[iters] += min_distance
            self.update_centroid()
        
    def cluster_label(self):
        self.cluster_acc = np.zeros(shape = self.k)
        #cluster_label = np.zeros(shape = self.k)
        for i in range(self.k):
            #Locate the index list of cluster i
            index = np.where(self.clusters == i)[0]
            #Total number of label in cluster i
            total_label_count = len(self.label[index])
            
            label_count_list = np.bincount(self.label[index].astype(np.int64))
            #target_label = np.where(label_count_list == i)[0]
            #target_label_count = np.where(label_count_list == i)[1]
            
            #cluster_label[i] = i
            self.cluster_acc[i] =  label_count_list[i]/total_label_count
        
        return np.mean(self.cluster_acc)
        
def plot_loss(input_list,label):
    plt.figure(figsize=(8,5))
    n = len(input_list)
    plt.plot(np.arange(0,n),input_list,color='red')
    plt.legend([label])
    plt.xticks(ticks = range(20), labels =  [i+1 for i in range(20)])
    plt.grid(True)
    plt.show() 

def plot_percentage(input_list,label,dimension_list):
    plt.figure(figsize=(8,5))
    n_dimension = len(dimension_list)
    plt.plot(np.arange(0,n_dimension),input_list,color='red')
    plt.legend([label])
    plt.xticks(ticks = range(n_dimension), labels =  dimension_list)
    plt.grid(True)
    plt.show() 


def plot_accuracy(dimension_list,train_acc, test_acc, train_acc_2 = [], test_acc_2 = []):
    plt.figure(figsize=(8,5))
    n_dimension = len(dimension_list)
    
    if train_acc_2 == []:
        plt.plot(np.arange(0,n_dimension),train_acc,color='red', linestyle ='-')
        plt.plot(np.arange(0,n_dimension),test_acc,color='red', linestyle ='--')
        plt.legend(['train acc','test acc'])
    else:
        plt.plot(np.arange(1,n_dimension),train_acc,color='red', linestyle ='-')
        plt.plot(np.arange(1,n_dimension),test_acc,color='red', linestyle ='--')
        plt.plot(np.arange(0,n_dimension),train_acc_2,color='green',linestyle ='-')
        plt.plot(np.arange(0,n_dimension),test_acc_2,color='green', linestyle ='--')
        plt.legend(['train acc','test acc','train acc-noise','test acc-noise'])
    plt.xticks(ticks = range(n_dimension), labels =  dimension_list)
    plt.grid(True)
    plt.show() 
dimension_list = [784,256,192,128,96,64,32,16,8]

df_train = pd.read_csv('mnist_train.csv', header = None )
df_test = pd.read_csv('mnist_test.csv', header = None)

np_train_data = df_train.to_numpy()
np_test_data = df_test.to_numpy()

train_label = np_train_data[:,0]
train_data = np_train_data[:,1:]

test_label = np_test_data[:,0]
test_data = np_test_data[:,1:]

X_train = train_data
y_train = train_label

X_test = test_data
y_test = test_label

#Task 2

train_acc_list = []
test_acc_list = []
for dimension in dimension_list:
    PCA_X_train, PCA_X_test = PCA(train_data = X_train,dimension = dimension, test_data= X_test)
    
    KNN_clf = KNeighborsClassifier(n_neighbors = 1)
    KNN_clf.fit(PCA_X_train,y_train)

    pred_train = KNN_clf.predict(PCA_X_train)
    pred_test = KNN_clf.predict(PCA_X_test)

    train_acc = accuracy_score(pred_train,y_train)
    test_acc = accuracy_score(pred_test,y_test)

    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

plot_accuracy(dimension_list,train_acc_list,test_acc_list)

#Task 3
Kmean = K_mean(k = 10)
Kmean.fit(train_data = X_train,label = y_train, n_iteration = 20,random_init = True)
plot_loss(Kmean.loss,'Loss')

#Task 4

result_list = []
for dimension in dimension_list:
    PCA_X_train, PCA_X_test = PCA(train_data = X_train,dimension = dimension, test_data= X_test)
    Kmean_PCA = K_mean(k = 10)
    Kmean_PCA.fit(train_data = PCA_X_train,label = y_train, n_iteration = 20,random_init = False)
    result_PCA = Kmean_PCA.cluster_label()
    result_list.append(result_PCA)
    #print(result_PCA)
    #print(Kmean_PCA.cluster_acc)
    #plot_loss(Kmean_PCA.loss)
plot_percentage(result_list,"Percentage",dimension_list)

#Task 5

train_mean = np.mean(X_train)
train_std = np.std(X_train)

train_noise = np.random.normal(loc = train_mean, scale = train_std, size = (X_train.shape[0],256))
X_train_with_noise = np.append(X_train, train_noise, axis = 1)

test_noise = np.random.normal(loc = train_mean, scale = train_std, size = (X_test.shape[0],256))
X_test_with_noise = np.append(X_test, test_noise, axis = 1)

train_acc_list = []
test_acc_list = []
train_acc_list_with_noise = []
test_acc_list_with_noise = []

noise_dimension_list = [1040,784,256,192,128,96,64,32,16,8]

for dimension in noise_dimension_list:
    PCA_train_with_noise, PCA_test_with_noise = PCA(train_data = X_train_with_noise,dimension = dimension, test_data= X_test_with_noise)
  
    KNN_clf_noise = KNeighborsClassifier(n_neighbors = 1)
    KNN_clf_noise.fit(PCA_train_with_noise,y_train)

    pred_train_noise = KNN_clf_noise.predict(PCA_train_with_noise)
    pred_test_noise = KNN_clf_noise.predict(PCA_test_with_noise)

    train_acc_noise = accuracy_score(pred_train_noise,y_train)
    test_acc_noise = accuracy_score(pred_test_noise,y_test)

    train_acc_list_with_noise.append(train_acc_noise)
    test_acc_list_with_noise.append(test_acc_noise)
    
    if dimension == 1040:
        pass
    else:
        PCA_train, PCA_test = PCA(train_data = X_train,dimension = dimension, test_data= X_test)
    
        KNN_clf = KNeighborsClassifier(n_neighbors = 1)
        KNN_clf.fit(PCA_train,y_train)

        pred_train = KNN_clf.predict(PCA_train)
        pred_test = KNN_clf.predict(PCA_test)

        train_acc = accuracy_score(pred_train,y_train)
        test_acc = accuracy_score(pred_test,y_test)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

plot_accuracy(noise_dimension_list,train_acc_list,test_acc_list,train_acc_list_with_noise,test_acc_list_with_noise)

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

parameters = {'C':[0.01,0.1,1,10,100]}
linear_svc = LinearSVC(random_state=0, max_iter = 100000)
clf = GridSearchCV(estimator = linear_svc, param_grid = parameters, cv = 5)
clf.fit(X_train,y_train)

train_acc_list = []
test_acc_list = []
train_acc_list_with_noise = []
test_acc_list_with_noise = []



for dimension in noise_dimension_list:
    PCA_train_with_noise, PCA_test_with_noise = PCA(train_data = X_train_with_noise,dimension = dimension, test_data= X_test_with_noise)
  
    best_svc_noise = clf.best_estimator_
    best_svc_noise.fit(PCA_train_with_noise,y_train)

    pred_train_noise = best_svc_noise.predict(PCA_train_with_noise)
    pred_test_noise = best_svc_noise.predict(PCA_test_with_noise)

    train_acc_noise = accuracy_score(pred_train_noise,y_train)
    test_acc_noise = accuracy_score(pred_test_noise,y_test)

    train_acc_list_with_noise.append(train_acc_noise)
    test_acc_list_with_noise.append(test_acc_noise)
    
    if dimension == 1040:
        pass
    else:
        PCA_train, PCA_test = PCA(train_data = X_train,dimension = dimension, test_data= X_test)
    
        best_svc = clf.best_estimator_
        best_svc.fit(PCA_train,y_train)

        pred_train = best_svc.predict(PCA_train)
        pred_test = best_svc.predict(PCA_test)

        train_acc = accuracy_score(pred_train,y_train)
        test_acc = accuracy_score(pred_test,y_test)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

plot_accuracy(noise_dimension_list,train_acc_list,test_acc_list,train_acc_list_with_noise,test_acc_list_with_noise)
"""


clf = KNeighborsClassifier(n_neighbors = 1)
clf.fit(X_train_with_noise,y_train)

pred_train = clf.predict(X_train_with_noise)
pred_test = clf.predict(X_test_with_noise)

train_acc = accuracy_score(pred_train,y_train)
test_acc = accuracy_score(pred_test,y_test)

print(f'train acc:{train_acc}')
print(f'test acc:{test_acc}')

PCA256_X_train, PCA256_X_test = PCA(train_data = X_train_with_noise,dimension = 128, test_data= X_test_with_noise)
print(PCA256_X_train.shape)
PCA256_clf = KNeighborsClassifier(n_neighbors = 1)
PCA256_clf.fit(PCA256_X_train,y_train)

PCA256_pred_train = PCA256_clf.predict(PCA256_X_train)
PCA256_pred_test = PCA256_clf.predict(PCA256_X_test)

PCA256_train_acc = accuracy_score(PCA256_pred_train,y_train)
PCA256_test_acc = accuracy_score(PCA256_pred_test,y_test)

print(f'PCA256 train acc:{PCA256_train_acc}')
print(f'PCA256 test acc:{PCA256_test_acc}')


result_list = []
for i in range(246):
    PCA_X_train, PCA_X_test = PCA(train_data = X_train,dimension = 256-i, test_data= X_test)
    Kmean_PCA = K_mean(k = 10)
    Kmean_PCA.fit(train_data = PCA_X_train,label = y_train, n_iteration = 10,random_init = False)
    result_PCA = Kmean_PCA.cluster_label()
    result_list.append(result_PCA)
    #print(result_PCA)
    #print(Kmean_PCA.cluster_acc)
    #plot_loss(Kmean_PCA.loss)
plot_loss(result_list)
"""
"""
clf = KNeighborsClassifier(n_neighbors = 1)
clf.fit(X_train,y_train)

pred_train = clf.predict(X_train)
pred_test = clf.predict(X_test)

train_acc = accuracy_score(pred_train,y_train)
test_acc = accuracy_score(pred_test,y_test)

print(f'train acc:{train_acc}')
print(f'test acc:{test_acc}')


PCA256_X_train, PCA256_X_test = PCA(train_data = X_train,dimension = 256, test_data= X_test)
print(PCA256_X_train.shape)
PCA256_clf = KNeighborsClassifier(n_neighbors = 1)
PCA256_clf.fit(PCA256_X_train,y_train)

PCA256_pred_train = clf.predict(PCA256_X_train)
PCA256_pred_test = clf.predict(PCA256_X_test)

PCA256_train_acc = accuracy_score(PCA256_pred_train,y_train)
PCA256_test_acc = accuracy_score(PCA256_pred_test,y_test)

print(f'PCA256 train acc:{PCA256_train_acc}')
print(f'PCA256 test acc:{PCA256_test_acc}')
"""