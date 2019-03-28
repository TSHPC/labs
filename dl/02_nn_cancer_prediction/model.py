#/usr/bin/env python3
import itertools 
import numpy             as np
import pandas            as pd 
import matplotlib.pyplot as plt
from sklearn               import preprocessing 
from sklearn.preprocessing import MinMaxScaler
from sklearn               import metrics
from sklearn.metrics       import confusion_matrix


class Model:
    def __init__(self, x, y):
        self.X      = x
        self.Y      = y
        self.Yh     = np.zeros((1, self.Y.shape[1]))
        self.L      = 2
        self.dims   = [9, 15, 1]
        self.params = {}
        self.ch     = {}
        self.grad   = {}
        self.loss   = []
        self.lr     = 0.003
        self.sam    = self.Y.shape[1]

    def init_params(self):
        np.random.seed(1)
        self.params['W1'] = np.random.randn(self.dims[1], self.dims[0]) / np.sqrt(self.dims[0])
        self.params['b1'] = np.zeros((self.dims[1], 1))        
        self.params['W2'] = np.random.randn(self.dims[2], self.dims[1]) / np.sqrt(self.dims[1]) 
        self.params['b2'] = np.zeros((self.dims[2], 1))                
        return

    def sigmoid(Z):
        return 1/(1+np.exp(-Z))

    def relu(Z):
        return np.maximum(0,Z)

    def d_relu(x):
        x[x<=0] = 0
        x[x>0]  = 1
        return x
    
    def d_sigmoid(Z):
        s  = 1/(1+np.exp(-Z))
        dZ = s * (1-s)
        return dZ
    
    def forward(self):    
        Z1 = self.params['W1'].dot(self.X) + self.params['b1'] 
        A1 = relu(Z1)
        self.ch['Z1'],self.ch['A1']=Z1,A1
        
        Z2 = self.params['W2'].dot(A1) + self.params['b2']  
        A2 = sigmoid(Z2)
        self.ch['Z2'],self.ch['A2']=Z2,A2
        self.Yh=A2
        loss=self.compute_loss(A2)
        return self.Yh, loss

    def compute_loss(self,Yh):
        loss = (1./self.sam) * (-np.dot(self.Y,np.log(Yh).T) - np.dot(1-self.Y, np.log(1-Yh).T))    
        return loss

    def backward(self):
        dLoss_Yh = - (np.divide(self.Y, self.Yh ) - np.divide(1 - self.Y, 1 - self.Yh))    
        
        dLoss_Z2 = dLoss_Yh * d_sigmoid(self.ch['Z2'])    
        dLoss_A1 = np.dot(self.param["W2"].T,dLoss_Z2)
        dLoss_W2 = 1./self.ch['A1'].shape[1] * np.dot(dLoss_Z2,self.ch['A1'].T)
        dLoss_b2 = 1./self.ch['A1'].shape[1] * np.dot(dLoss_Z2, np.ones([dLoss_Z2.shape[1],1])) 
                            
        dLoss_Z1 = dLoss_A1 * d_relu(self.ch['Z1'])        
        dLoss_A0 = np.dot(self.param["W1"].T,dLoss_Z1)
        dLoss_W1 = 1./self.X.shape[1] * np.dot(dLoss_Z1,self.X.T)
        dLoss_b1 = 1./self.X.shape[1] * np.dot(dLoss_Z1, np.ones([dLoss_Z1.shape[1],1]))  
        
        self.params["W1"] = self.params["W1"] - self.lr * dLoss_W1
        self.params["b1"] = self.params["b1"] - self.lr * dLoss_b1
        self.params["W2"] = self.params["W2"] - self.lr * dLoss_W2
        self.params["b2"] = self.params["b2"] - self.lr * dLoss_b2

    def gd(self, X, Y, iter = 3000):
        np.random.seed(1)                         
    
        self.init_params()
    
        for i in range(0, iter):
            Yh, loss=self.forward()
            self.backward()
            if i % 500 == 0:
                print ("Cost after iteration %i: %f" %(i, loss))
                self.loss.append(loss)
        return

def main():
    x = np.random.randn(10, 10)
    y = np.random.randn(10, 10)
    m = Model(x,y)
    m.init_params()
    m.forward()

if __name__ == '__main__':
    main()
