# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:11:50 2019

@author: Master
"""
import random
import numpy
import matplotlib.pyplot as plt
D1=[0.697,0.774,0.634,0.608,0.556,0.403,0.481,0.437,0.666,0.243,0.245,0.343,0.639,0.657,0.360,0.593,0.719]
D2=[0.460,0.376,0.264,0.318,0.215,0.237,0.149,0.211,0.091,0.267,0.057,0.099,0.161,0.198,0.370,0.042,0.103]
YT=[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]
d=2 #number of input neurons
q=4 #number of hidden neurons
l=3 #number of output neurons
Eta=0.01 #learning rate
X=numpy.zeros((1,d)) #input neurons
V=numpy.ones((d+1,q)) #input weight
H=numpy.zeros((1,q)) #input of hidden neurons
Theta=numpy.zeros((1,q)) #hidden neurons threshold
HT=numpy.ones((1,q+1)) #hidden neurons output
E=numpy.zeros((1,q)) #coefficients of backward from hidden units
W=numpy.ones((q+1,l)) #hidden units weight
YI=numpy.zeros((1,l)) #input of output neurons
Gamma=numpy.zeros((1,l)) #output neurons threshold
YP=numpy.zeros((len(D1),l)) #output neurons
g=numpy.ones((1,l)) #backward coefficients
Error=numpy.ones((1,len(D1)))
OldError=numpy.ones((1,len(D1)))
# initialize all the parameters with random numbers
for i in X:
    i=random.uniform(0,10)
for i in V:
    i=random.uniform(0,10)
for i in Theta:
    i=0.5
for i in W:
    i=random.uniform(0,10)
# train MLP
i0=1
while i0<100000 :
    for k in range(len(D1)):
        for j in range(q):
            H[0,j]=D1[k]*V[0,j]+D2[k]*V[1,j]+V[2,j]
            HT[0,j]=1/(1+numpy.exp(Theta[0,j]-H[0,j]))
        Error[0,k]=0
        for i in range(l):
            YI[0,i]=0
            for j in range(q+1):
                YI[0,i]=YI[0,i]+W[j,i]*HT[0,j]
            YP[k,i]=1/(1+numpy.exp(Gamma[0,i]-YI[0,i]))
            g[0,i]=YP[k,i]*(1-YP[k,i])*(YT[k]-YP[k,i])
            Error[0,k]=Error[0,k]+(YT[k]-YP[k,i])*(YT[k]-YP[k,i])/l
        for i in range(l):
            for j in range(q+1):
                W[j,i]=W[j,i]+Eta*g[0,i]*HT[0,j]
        for i in range(l):
            Gamma[0,i]=Gamma[0,i]-Eta*g[0,i]
        for j in range(q):
            E[0,j]=0
            for i in range(l):
                E[0,j]=E[0,j]+HT[0,j]*(1-HT[0,j])*W[j,i]*g[0,i]
        for j in range(q):
            V[0,j]=V[0,j]+Eta*E[0,j]*D1[k]
            V[1,j]=V[1,j]+Eta*E[0,j]*D2[k]
            Theta[0,j]=Theta[0,j]-Eta*E[0,j]
    print(numpy.mean(Error))
    i0=i0+1
YP=numpy.mean(YP,1)
plt.plot(YT)
plt.plot(YP)
plt.show()
    


            