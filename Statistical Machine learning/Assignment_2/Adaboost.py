import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import copy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

class Adaboost:
    def __init__(self,weak_clf):
        self.weights = None
        self.alphas = None
        self.stumps = []
        self.errors = None
        self.weak_clf = weak_clf

    def fit(self,X,y,iters):
        number_of_data = len(y)
        self.weights = np.zeros(shape=(iters, number_of_data))
        self.alphas = np.zeros(shape=iters)
        self.errors = np.zeros(shape=iters)
        self.weights[0] = 1/number_of_data
       

        for t in range(iters):
    
            weight = self.weights[t]
            temp_clf = copy.deepcopy(self.weak_clf)
            temp = temp_clf.fit(X,y,sample_weight = weight)
            self.stumps.append(temp)

            pred = self.stumps[t].predict(X)
   
            error = np.sum(weight[pred != y])
            self.errors[t] = error
            self.alphas[t] = np.log((1-error)/error)/2

            new_weight = weight*np.exp(-self.alphas[t]*y*pred)/2
            #print(new_weight.sum())
            new_weight /= new_weight.sum()  # normalize the weight
            #print(self.weights[t])
            
            if t+1 == iters:
                pass
            else:
                self.weights[t+1] = new_weight
    
    def predict(self, input_X):
        
        h_t = np.array([stump.predict(input_X) for stump in self.stumps])
        H_t = np.sign(np.dot(self.alphas,h_t))
        return H_t
            
def accuracy(pred,target):
    accuracy = np.sum(pred==target)/len(target)
    return accuracy

def plot_graph(list_1,list_2,legend_1,legend_2):
    plt.figure(figsize=(8,5))
    n = len(list_1)
    plt.plot(np.arange(0,n),list_1,color='orange')
    plt.plot(np.arange(0,n),list_2,color='b')
    plt.legend([legend_1,legend_2])
        
    plt.grid(True)
    plt.show() 
        
df_data = pd.read_csv('wdbc_data.csv',header = None)
df_data.replace('M',-1,inplace = True)
df_data.replace('B',1,inplace = True)
df_data.drop([0], axis=1,inplace =  True)

np_data = np.asarray(df_data)

data = np_data[:,1:]
target = np_data[:,0]

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.4727, shuffle = False)

number_of_iteration = 200
    
start = time.time()
clf = AdaBoostClassifier(n_estimators = number_of_iteration,algorithm = 'SAMME',base_estimator = DecisionTreeClassifier(max_depth=1,max_leaf_nodes=2))
clf.fit(X_train,y_train)
end = time.time()

sk_train_time = end - start

start = time.time()
sk_train_pred = clf.predict(X_train)
end = time.time()

sk_test_time = end - start

sk_train_acc  = accuracy(sk_train_pred,y_train)
sk_train_err = 1 - sk_train_acc
sk_test_pred = clf.predict(X_test)
sk_test_acc  = accuracy(sk_test_pred,y_test)
sk_test_err = 1 - sk_test_acc

print(f'sklearn train_acc:{sk_train_acc}')
print(f'sklearn test_acc:{sk_test_acc}')
print(f'sklearn train time:{sk_train_time}')
print(f'sklearn test time:{sk_test_time}')

start = time.time()
adaboost = Adaboost(DecisionTreeClassifier(max_depth=1,max_leaf_nodes=2))
adaboost.fit(X_train,y_train,number_of_iteration)
end = time.time()
adaboost_train_time = end - start

start = time.time()
train_pred = adaboost.predict(X_train)
end = time.time()
adaboost_test_time = end - start

train_acc  = accuracy(train_pred,y_train)
train_err = 1 - train_acc
test_pred = adaboost.predict(X_test)
test_acc  = accuracy(test_pred,y_test)
test_err = 1 - test_acc

print(f'train_acc:{train_acc}')
print(f'test_acc:{test_acc}')
print(f'train time:{adaboost_train_time}')
print(f'test time:{adaboost_test_time}')

plot_graph(clf.estimator_weights_,adaboost.alphas,'sklearn alpha','adaboost alpha')
plot_graph(clf.estimator_errors_,adaboost.errors,'sklearn error','adaboost erorr')


kernel_list = ['linear','poly', 'rbf']

for kernel in kernel_list:
    svc_clf = SVC(kernel = kernel,C = 1)
    start = time.time()
    svc_clf.fit(X_train,y_train)
    end = time.time()

    svc_train_time = end - start

    start = time.time()
    svc_train_pred = svc_clf.predict(X_train)
    end = time.time()
    svc_test_time = end - start


    svc_train_acc  = accuracy(svc_train_pred,y_train)
    svc_train_err = 1 - svc_train_acc
    svc_test_pred = svc_clf.predict(X_test)
    svc_test_acc  = accuracy(svc_test_pred,y_test)
    svc_test_err = 1 - svc_test_acc

    print(f'{kernel} svc_train_acc:{svc_train_acc}')
    print(f'{kernel} svc_test_acc:{svc_test_acc}')
    print(f'{kernel} svc train time:{svc_train_time}')
    print(f'{kernel} svc test time:{svc_test_time}')

train_err_list = [[],[],[]]
test_err_list = [[],[],[]]

for depth in range(3):
        
    for i in range(number_of_iteration):
        adaboost = Adaboost(DecisionTreeClassifier(max_depth=depth+1))
        adaboost.fit(X_train,y_train,i+1)

        train_pred = adaboost.predict(X_train)
        train_acc  = accuracy(train_pred,y_train)
        train_err = 1 - train_acc
        test_pred = adaboost.predict(X_test)
        test_acc  = accuracy(test_pred,y_test)
        test_err = 1 - test_acc
        train_err_list[depth].append(train_err)
        test_err_list[depth].append(test_err)
    plot_graph(train_err_list[depth],test_err_list[depth],'Train error','Test erorr')


train_err_list = [[],[],[]]
test_err_list = [[],[],[]]

for k, kernel in enumerate(kernel_list):      
    for i in range(10):
        adaboost = Adaboost(SVC(kernel = kernel,C = 1))
        adaboost.fit(X_train,y_train,i+1)

        train_pred = adaboost.predict(X_train)
        train_acc  = accuracy(train_pred,y_train)
        train_err = 1 - train_acc
        test_pred = adaboost.predict(X_test)
        test_acc  = accuracy(test_pred,y_test)
        test_err = 1 - test_acc
        train_err_list[k].append(train_err)
        test_err_list[k].append(test_err)
    plot_graph(train_err_list[k],test_err_list[k],'Train error','Test erorr')




