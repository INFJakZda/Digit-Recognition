import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os
# print(os.listdir("../data"))

# relu activation function
def relu(x):
    return x*(x>0)

# hypothesis function: simple FNN with 1 hidden layer
def h(X,W,b):
    # layer 1 = input layer
    a1 = X
    # layer 1 (input layer) -> layer 2 (hidden layer)
    z1 = np.matmul(X, W[0]) + b[0]
    # layer 2 activation
    a2 = relu(z1)
    # layer 2 (hidden layer) -> layer 3 (output layer)
    z2 = np.matmul(a2, W[1])
    s = np.exp(z2)
    total = np.sum(s, axis=1).reshape(-1,1)
    sigma = s/total
    # the output is a probability for each sample
    return sigma

# activation function for the last FC layer: softmax function 
# Output: K probabilities represent an estimate of P(y=k|X_in;weights) for k=1,...,K
def softmax(X_in,weights):
    # the weights has shape (n, K)
    # n: the number of features X_in has
    # n = X_in.shape[1]
    # K: the number of classes
    # K = 10
    s = np.exp(np.matmul(X_in,weights))
    total = np.sum(s, axis=1).reshape(-1,1)
    return s / total

# loss function: cross entropy with L^2 regularization
def loss(W,b,X,y,alpha=1e-4):
    # N: number of samples in the batch
    K = 10
    N = X.shape[0]
    # KL_sample stores the cross entropy for each sample in X
    KL_sample = np.zeros(N)
    for k in range(K):
        KL_sample += np.log(h(X,W,b))[:,k] * (y == k)
    # KL_sample is a dimension (N,) array
    reg = 0.5*alpha*(np.sum(W[1]**2) + np.sum(W[0]**2))
    return -np.mean(KL_sample) + reg

def backprop(W,b,X,y,alpha=1e-4):
    '''
    Step 1: explicit forward pass h(X;W,b)
    Step 2: backpropagation for dW and db
    '''
    K = 10
    N = X.shape[0]
    
    ### Step 1:
    # layer 1 = input layer
    a1 = X
    # layer 1 (input layer) -> layer 2 (hidden layer)
    z1 = np.matmul(X, W[0]) + b[0]
    # layer 2 activation
    a2 = relu(z1)
    # layer 2 (hidden layer) -> layer 3 (output layer)
    z2 = np.matmul(a2, W[1])
    s = np.exp(z2)
    total = np.sum(s, axis=1).reshape(-1,1)
    sigma = s/total
    
    ### Step 2:
    # layer 2->layer 3 weights' derivative
    # delta2 is \partial L/partial z2
    delta2 = np.zeros([N,K])
    for k in range(K):
        delta2[:,k] = (sigma[:,k] - (y==k))
    grad_W1 = np.matmul(a2.T, delta2)
    # layer 1->layer 2 weights' derivative
    # delta1 is \partial a2/partial z1
    # layer 2 activation's (weak) derivative is 1*(z1>0)
    delta1 = np.matmul(delta2, W[1].T)*(z1>0)
    grad_W0 = np.matmul(X.T, delta1)
    # no derivative for layer 1
    
    
    dW = [grad_W0/N + alpha*W[0], grad_W1/N + alpha*W[1]]
    db = [np.mean(delta1, axis=0)]
    # dW[0] is W[0]'s derivative, and dW[1] is W[1]'s derivative; similar for db
    return dW, db

if __name__ == '__main__':
    # Read the data
    train_data = pd.read_csv('../data/train.csv')
    test_data = pd.read_csv("../data/test.csv")

    # Set up the data
    y_train = train_data['label'].values
    X_train = train_data.drop(columns=['label']).values/255
    X_test = test_data.values/255


    # hyper-parameters and network
    eta = 5e-1
    alpha = 1e-6 # regularization
    gamma = 0.99 # RMSprop
    eps = 1e-3 # RMSprop
    num_iter = 1002 # number of iterations of gradient descent
    n_H = 256 # number of neurons in the hidden layer
    n = X_train.shape[1] # number of pixels in an image
    K = 10

    # initialization
    np.random.seed(42)
    W = [1e-1*np.random.randn(n, n_H), 1e-1*np.random.randn(n_H, K)]
    b = [np.random.randn(n_H)]

    gW0 = gW1 = gb0 = 1

    loss_list = []
    accuracy_list = []

    for i in range(num_iter):
        dW, db = backprop(W,b,X_train,y_train,alpha)
        
        gW0 = gamma*gW0 + (1-gamma)*np.sum(dW[0]**2)
        etaW0 = eta/np.sqrt(gW0 + eps)
        W[0] -= etaW0 * dW[0]
        
        gW1 = gamma*gW1 + (1-gamma)*np.sum(dW[1]**2)
        etaW1 = eta/np.sqrt(gW1 + eps)
        W[1] -= etaW1 * dW[1]
        
        gb0 = gamma*gb0 + (1-gamma)*np.sum(db[0]**2)
        etab0 = eta/np.sqrt(gb0 + eps)
        b[0] -= etab0 * db[0]
        
        # if i % 10 == 0:
        #     loss_list.append(loss(W,b,X_train,y_train,alpha))
        #     accuracy_list.append(np.mean(np.argmax(h(X_train,W,b), axis=1)== y_train))

        if i % 500 == 0:
            #Vote errors
            # x = np.arange(0, len(loss_list))
            # y = [ loss_list, accuracy_list ]
            # labels = ['Loss', 'Accuracy']

            # plt.xlabel('Iterations')
            # plt.ylabel('Loss / Accuracy [%]')

            # for y_arr, label in zip(y, labels):
            #     plt.plot(x, y_arr, label=label)
            # plt.axis([1, len(loss_list), 0, 100])
            # plt.legend()
            # plt.show()

            # sanity check 1
            print("loss after", i+1, "iterations is {:.8}".format(
                loss(W,b,X_train,y_train,alpha)))
            print("training accuracy after", i+1, "iterations is {:.4%}".format( 
                np.mean(np.argmax(h(X_train,W,b), axis=1)== y_train)))
            
            # sanity check 2
            # print("gW0={:.4f} gW1={:.4f} gb0={:.4f}\netaW0={:.4f} etaW1={:.4f} etab0={:.4f}"
            #     .format(gW0, gW1, gb0, etaW0, etaW1, etab0))
            
            # sanity check 3
            # print("|dW0|={:.5f} |dW1|={:.5f} |db0|={:.5f}"
            #     .format(np.linalg.norm(dW[0]), np.linalg.norm(dW[1]), np.linalg.norm(db[0])), "\n")
            
            # reset RMSprop
            gW0 = gW1 = gb0 = 1

    print("final loss is {:.8}".format(loss(W,b,X_train,y_train,alpha)))
    print("final training accuracy is {:.4%}".format(np.mean(np.argmax(h(X_train,W,b), axis=1)== y_train)))

    # predictions
    y_pred = np.argmax(h(X_test,W,b), axis=1)

    # Generating submission using pandas for grading
    submission = pd.DataFrame({'ImageId': range(1,len(X_test)+1) ,'Label': y_pred })
    submission.to_csv("simplemnist_result.csv",index=False)
