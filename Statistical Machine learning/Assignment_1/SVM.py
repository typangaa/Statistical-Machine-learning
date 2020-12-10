import numpy as np
import pandas as pd
import time
from numpy.linalg import matrix_rank
from cvxopt import matrix
from cvxopt import solvers
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

def svm_train_primal(data_train,label_train,regularisation_para_C):

    X = data_train
    y = label_train.reshape(len(X),1)

    n_data = data_train.shape[0]
    n_parameter = data_train.shape[1]
    len_slack = n_data

    np_P = np.diag(np.ones(n_parameter+1+len_slack))
    np_P[:,n_parameter:] = 0 
    #print(np_P.shape)
    #print(matrix_rank(np_P))

    np_q1 = np.zeros((n_parameter+1,1))
    np_q2 = (float(regularisation_para_C)/n_data)*np.ones((len_slack,1))
    #print(np_q1.shape)
    #print(np_q2.shape)
    np_q = np.vstack((np_q1,np_q2))
    #print(np_q.shape)


    X_with_bias = np.hstack((X, np.ones((n_data,1))))
    np_G1 = X_with_bias*y*-1
    #print(np_G1)
    np_G2 = np.diag(np.ones(len_slack))*-1
    #np_G2 = np.ones((n_data,len_slack))*-1
    np_G3 = np.zeros((n_data,n_parameter+1))
    np_G4 = np.diag(np.ones(len_slack))*-1

    temp_g1 = np.hstack((np_G1, np_G2))
    temp_g2 = np.hstack((np_G3, np_G4))
    #print(np_G1.shape)
    #print(np_G2.shape)
    np_G = np.vstack((temp_g1, temp_g2))
    #print(np_G.shape)
    #print(matrix_rank(np_G))
    #print(np_G)

    np_h1 = (-1) * np.ones((n_data,1))
    np_h2 =  np.zeros((n_data,1))
    #print(np_h1.shape)
    np_h = np.vstack((np_h1, np_h2))
    #print(np_h.shape)


    P = matrix(np_P, tc='d')
    q = matrix(np_q, tc='d')
    G = matrix(np_G, tc='d')
    h = matrix(np_h, tc='d')

    sol = solvers.qp(P,q,G,h)

    return sol['x'][0:n_parameter+1]

def svm_predict_primal(data_test,label_test,svm_model):
    n_data = data_test.shape[0]
    n_parameter = data_test.shape[1]
    w = np.asarray(svm_model[0:n_parameter])
    b = svm_model[n_parameter]
    
    predict_y = np.sign(np.dot(data_test,w,)+b)

    label_test = label_test.reshape(n_data,1)
    #print(label_test.shape)
    accuracy_list = predict_y==label_test
    #print(accuracy_list)
    accuracy = np.sum(accuracy_list)/n_data

    return accuracy

def svm_train_dual(data_train,label_train,regularisation_para_C):

    X = data_train
    y = label_train.reshape(len(X),1)
    
    n_data = data_train.shape[0]
    n_parameter = data_train.shape[1]
    
    np_P = (y*X).dot((y*X).T)
    #print(np_P.shape)
   
    np_q = -1*np.ones(n_data)
    #print(np_q.shape)
    
    np_G1 = -1*np.diag(np.ones(n_data))
    np_G2 = np.diag(np.ones(n_data))
    #print(np_G1.shape)
    #print(np_G2.shape)
    np_G = np.vstack((np_G1, np_G2))
    #print(np_G.shape)
    
    np_h1 = np.zeros((n_data,1))
    np_h2 = (regularisation_para_C/n_data) * np.ones((n_data,1))
    #print(np_h1.shape)
    #print(np_h2.shape)
    np_h = np.vstack((np_h1, np_h2))
    #print(np_h.shape)

    np_A = y.T
    #print(np_A.shape)

    np_b = 0

    P = matrix(np_P, tc='d')
    q = matrix(np_q, tc='d')
    G = matrix(np_G, tc='d')
    h = matrix(np_h, tc='d')
    A = matrix(np_A, tc='d')
    b = matrix(np_b, tc='d')

    sol = solvers.qp(P,q,G,h,A,b)
    #print(sol['x']) 
    
    alphas = sol['x']
    np_alphas = np.asarray(alphas)
    
    W = X_train.T.dot(alphas*y_train.reshape(len(y_train),1))
    #print(W.shape)
    #print(np_alphas)
    #Select the support vector with 0 < alpah value < C/n 
    S = np.logical_and((np_alphas < regularisation_para_C/n_data - 1e-4), (np_alphas > 1e-8)).flatten()
    #print(S)
    b = np.average(y[S] - np.dot(X_train[S], W))
    #print(b)

    output = np.concatenate((W,b.reshape((1,1))),0)

    return output

def svm_predict_dual(data_test,label_test,svm_model):
    n_data = data_test.shape[0]
    n_parameter = data_test.shape[1]
    w = np.asarray(svm_model[0:n_parameter])
    b = svm_model[n_parameter]

    predict_y = np.sign(np.dot(data_test,w,)+b)

    label_test = label_test.reshape(n_data,1)
    #print(label_test.shape)
    accuracy_list = predict_y==label_test
    #print(accuracy_list)
    accuracy = np.sum(accuracy_list)/n_data

    return accuracy

def weight_bias_different(W_vector_1,W_vector_2):

    weight_d = np.sqrt(np.sum(np.square(W_vector_1[:-1]-W_vector_2[:-1])))
    bias_d = np.abs(W_vector_1[-1]-W_vector_2[-1])

    return weight_d, bias_d 
    

train_set  = pd.read_csv('train.csv',header=None)
np_train_set = train_set.to_numpy()

test_set  = pd.read_csv('test.csv',header=None)
np_test_set = test_set.to_numpy()

train_X = np_train_set[:,1:]
train_y = np_train_set[:,0]
train_y[train_y==0] = -1

X_test = np_test_set[:,1:]
y_test = np_test_set[:,0]
y_test[y_test==0] = -1

X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.2, random_state=1)

n_data = X_train.data.shape[0]
n_parameter = X_train.data.shape[1]

param_grid = {'C': [0.1/n_data,0.3/n_data,1/n_data,3/n_data,10/n_data,33/n_data,100/n_data,333/n_data,1000/n_data]}

grid_search_clf = SVC(kernel = 'linear')
# run grid search
#t3 = time.time()
grid_search = GridSearchCV(grid_search_clf, param_grid=param_grid, cv = 5)
grid_search.fit(X_train, y_train)
#t4 = time.time()
#print(t4-t3)

df_result = pd.DataFrame(grid_search.cv_results_)
#print(df_result.sort_values(by=['rank_test_score']).head(10))
best_C = df_result[df_result['rank_test_score']==1]['param_C']
best_C = np.asarray(best_C,dtype='float')
#print(best_C.shape)
#print(best_C*n_data)

C_parameter = best_C*n_data

# train the svm in the sklearn
clf = SVC(kernel = 'linear', C = C_parameter/n_data)
t1 = time.time()
clf.fit(X_train, y_train)
t2 = time.time()
print(f'Run time on skelarn:{t2-t1}')
print('w = ',clf.coef_)
print('b = ',clf.intercept_)

# train the svm in the primal
t3 = time.time()
w_primal = svm_train_primal(X_train,y_train,C_parameter)
t4 = time.time()
print(f'Run time on primal:{t4-t3}')

# train the svm in the dual
t5 = time.time()
w_dual = svm_train_dual(X_train,y_train,C_parameter)
t6 = time.time()
print(f'Run time on dual:{t6-t5}')

test_accuracy_primal = svm_predict_primal( X_test , y_test , w_primal )
val_accuracy_primal = svm_predict_primal( X_val , y_val , w_primal )
train_accuracy_primal = svm_predict_primal( X_train , y_train , w_primal )
print('Primal')
print(f'train acc:{train_accuracy_primal}')
print(f'val acc:{val_accuracy_primal}')
print(f'test acc:{test_accuracy_primal}')


test_accuracy_dual = svm_predict_dual( X_test , y_test , w_dual )
val_accuracy_dual = svm_predict_dual( X_val , y_val , w_dual )
train_accuracy_dual = svm_predict_dual( X_train , y_train , w_dual )
print('Dual')
print(f'train acc:{train_accuracy_dual}')
print(f'val acc:{val_accuracy_dual}')
print(f'test acc:{test_accuracy_dual}')


sklearn_w = np.concatenate((np.asarray(clf.coef_),np.asarray(clf.intercept_).reshape((1,1))),1).T
#print(sklearn_w.shape)

test_accuracy = svm_predict_primal( X_test , y_test , sklearn_w)
val_accuracy = svm_predict_primal( X_val , y_val , sklearn_w)
train_accuracy = svm_predict_primal( X_train , y_train , sklearn_w)
print('sklearn')
print(f'train acc:{train_accuracy}')
print(f'val acc:{val_accuracy}')
print(f'test acc:{test_accuracy}')


w_d,b_d = weight_bias_different(w_primal,w_dual)
print(f'weight square sum different:{w_d},bias different:{b_d}')
w_d,b_d = weight_bias_different(w_primal,sklearn_w)
print(f'weight square sum different:{w_d},bias different:{b_d}')
w_d,b_d = weight_bias_different(w_dual,sklearn_w)
print(f'weight square sum different:{w_d},bias different:{b_d}')