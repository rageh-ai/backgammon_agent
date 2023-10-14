import numpy as np
import pandas as pd
import scipy.stats as stats
import sys

class ml_perceptron:

    def sigmoid(self, x):

        return (1/(1+np.exp(-x)))


    def passthrough(self, feat_vector):
        feat_vector = np.array(feat_vector).reshape((16,1))
        feat_vector = self.normalization(feat_vector)

        self.pass_1 = np.dot(self.w_1, feat_vector) + self.b_1
        self.pass_1_sig = self.sigmoid(self.pass_1) 

        self.pass_2 = np.dot(self.w_2, self.pass_1_sig) + self.b_2
        self.pass_2_sig = self.sigmoid(self.pass_2)



        return (self.pass_2_sig)


    def update(self,new_v, state_t, reward=0):
    
        discount = 0.5
        lr = 0.7
        lamda = 0.6

        state_t = np.array(state_t).reshape((16,1))
        self.passthrough(state_t)

        loss = reward + new_v - self.v_t

        v_w2 = self.sigmoid_derivative(self.pass_2) * self.pass_1_sig.T
        v_b2 = self.sigmoid_derivative(self.pass_2)

        v_w1 = np.multiply((self.sigmoid_derivative(self.pass_2) * self.w_2.T), self.sigmoid_derivative(self.pass_1)) * state_t.T
        v_b1 = np.multiply(np.multiply(self.sigmoid_derivative(self.pass_2), self.w_2.T), self.sigmoid_derivative(self.pass_1))

        self.z_t_w1 = (discount * self.z_t_w1) +  (lamda) * v_w1
        self.z_t_w2 = (discount *  self.z_t_w2) + (lamda) * v_w2

        self.z_t_b2 = (discount * self.z_t_b2) + (lamda) * v_b2
        self.z_t_b1 = (discount * self.z_t_b1) + (lamda) * v_b1


        self.w_2 = self.w_2 + (lr * loss * self.z_t_w2)
        self.w_1 = self.w_1 + (lr * loss * self.z_t_w1)

        self.b_2 = self.b_2 + (lr * loss * self.z_t_b2)
        self.b_1 = self.b_1 + (lr * loss * self.z_t_b1)

        self.v_t = new_v
        self.time+=1

        
    def sigmoid_derivative(self,z):
        return np.multiply(self.sigmoid(z), (1 - self.sigmoid(z)))

    def normalization(self,X):
        X = X.astype(np.float32)
        for val in range(X.shape[0]):
            if val < 12:
                X[val] = X[val]/15
            if val == 12:
                X[val] = X[val]/15
            if val == 13:
                X[val] = X[val]/15

        return X
        

    def __init__(self, inp, out, hidden): 
        self.input_no = inp
        self.output_no = out
        self.hidden_no = hidden

        self.w_1 = np.random.uniform(low = -1/np.sqrt(inp) , high= 1/np.sqrt(inp), size=(self.hidden_no,self.input_no)) 
        self.w_2 = np.random.uniform(low = -1/np.sqrt(hidden),high = 1/np.sqrt(hidden), size=(self.output_no, self.hidden_no)) 

        self.b_1 = np.zeros((self.hidden_no,1))
        self.b_2 = np.zeros((self.output_no,1))
        
        self.v_t = 0.5
        self.z_t_w1 = 0
        self.z_t_w2 = 0
        self.z_t_b1 = 0
        self.z_t_b2 = 0
        self.time = 0
        self.exp = 0






        
        
    

