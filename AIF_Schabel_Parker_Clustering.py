#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 11:41:23 2022

@author: Yohn Taylor
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate 
from matplotlib.pyplot import cm
import random
from sklearn.cluster import KMeans
from scipy.optimize import least_squares
from scipy import optimize
from concentrationFunc_AIF import AIF_Fun_SP
from random import randint


# Extended tofts integral function
def Extended_Tofts_Integral(t, Cp, Kt=0.1, ve=0.2, vp=0.1, uniform_sampling=True):
    nt = len(t)
    Ct = np.zeros(nt)
    for k in range(nt):
        tmp = vp*Cp[:k+1] + integrate.cumtrapz(np.exp(-Kt*(t[k]-t[:k+1])/ve)*Cp[:k+1],t[:k+1], initial=0.0)
        # print(tmp)
        Ct[k] = tmp[-1]
        # print(Ct[k])
    return Ct


''' Parameters associated with the Tofts model'''

# AIF parameter values Schabel et al.

Ktrans = 0.1
Ve = 0.3
Vp = 0.04
Kep = Ktrans/Ve
Hct = 0.42

# Functional form Schabel and Parker

A0 = 0.8152
A1 = 5.8589
A2 = 0.9444
A3 = 0.4888

Delta0 = 0.1563
Delta1 = Delta0
Delta2 = Delta0
Delta3 = Delta0

a0 = 7.9461
a1 = 2.5393
a2 = a0
a3 = a0

tau0 = 0.14
tau1 = 0.04286
tau2 = 0.06873
tau3 = tau0

T = 9.6319

''' Time increments (mins)'''

M = 100
N = 2
t = np.linspace(0,N,M)


Param = np.array([A0,A1,A2,A3,Delta0,a0,a1,tau0,tau1,tau2,T])
    
def Extended_Tofts_Integral2(t, A0,A1,A2,A3,Delta0,a0,a1,tau0,tau1,tau2,T, Kt, ve, vp, uniform_sampling=True):
    # Param = [A0,A1,A2,A3,Delta0,a0,a1,a2,tau0,tau1,tau2,T]
    Cp = AIF_Fun_SP(A0,A1,A2,A3,Delta0,a0,a1,tau0,tau1,tau2,T, t)       
    nt = len(t)
    Ct = np.zeros(nt)
    for k in range(nt):
        tmp = vp*Cp[:k+1] + integrate.cumtrapz(np.exp(-Kt*(t[k]-t[:k+1])/ve)*Cp[:k+1],t[:k+1], initial=0.0)
        # print(tmp)
        Ct[k] = tmp[-1]
        # print(Ct[k])
    return Ct


AIF = AIF_Fun_SP(*Param, t)  
plt.figure(1)
plt.plot(t, AIF)
plt.ylabel("AIF (mM)")
plt.xlabel("Time (s)")


# Uniform random distribution of the kinetic parameters
Num = 1000
Ktrans = np.random.uniform(0,0.5,Num)
Ve = np.random.uniform(0.15,1,Num)
Vp = np.random.uniform(0,0.2,Num)


# Adding guassian noise (mM) with mean = 0, and Std = 0.08e-3 M
noise = np.random.normal(0,0.08,M) 


# Plotting the generated random uniform distributions
plt.figure(2)
plt.subplot(311)
plt.title("Random uniform distribution of kinetic parameters")
plt.xlabel("Ktrans")
plt.ylabel("Count")

plt.hist(Ktrans, bins=np.arange(0,0.5,0.005), edgecolor='blue') 
plt.subplot(312)
plt.hist(Ve, bins=np.arange(0.15,1,0.0075), edgecolor='blue') 
plt.xlabel("Ve")
plt.ylabel("Count")

plt.subplot(313)
plt.hist(Vp, bins=np.arange(0,0.20,0.0025), edgecolor='blue') 
plt.xlabel("Vp")
plt.ylabel("Count")

plt.tight_layout()
plt.show()


# Generating the different curves with the randomly distributed kinetic parameters
ConcTiss = np.zeros((Num,M))

for i in range(1,Num):
    ConcTiss[i,:] = Extended_Tofts_Integral(t, AIF, Kt=Ktrans[i], ve=Ve[i], vp=Vp[i], uniform_sampling=True) + noise  

plt.figure(3)
plt.plot(t, ConcTiss[2,:],'.')
plt.ylabel("Total tissue concentraion $C_t$ (mM)")
plt.xlabel("Time (min)")


# Extract 10 sets of four curves from the data sample
Ten_sets_of_four = np.zeros((40,M))

# Arbitrary placeholder
a = 0

for i in range(1,len(Ten_sets_of_four)):
    Ten_sets_of_four[i,:] = ConcTiss[a,:]
    a = i*10
    
plt.figure(4)
plt.plot(t, Ten_sets_of_four[2,:],'o')
    


# Cluster the data from the extracted tissue concentration curves

''' Elbow method empirically determines how many clusters are needed. Num of Clusters > 5 '''


# Elbow method
cs = []
for i in range(3, 40):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(Ten_sets_of_four)
    cs.append(kmeans.inertia_)
    
plt.figure(5)
plt.plot(range(3, 40), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()




# Clustering method
kmeans = KMeans(n_clusters = 8, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
kmeans.fit(Ten_sets_of_four)
labels = kmeans.predict(Ten_sets_of_four)
# print(kmeans.predict(Ten_sets_of_four))
# print(labels)

filtered_label = Ten_sets_of_four[labels == 1]
color = cm.rainbow(np.linspace(0, 1, 8))


filtered_label=[]
averageCurve = []
for i in range(0,8):
    filtered_label0 = Ten_sets_of_four[labels == i]
    filtered_label.append(filtered_label0)
    clustered_average = []
    for j in range(len(filtered_label[i])):
        
        clusters = filtered_label[i][j]
        plt.figure(1)
        plt.plot(t, clusters, c = color[i])
        clustered_average.append(clusters)
        plt.xlabel('time(mins)')
        plt.ylabel(' $C_t$ (mM)')
        plt.title("Multiple tissue concentration curve clusters")
        a = np.asarray(clustered_average)       
    averageCurve_app = a.mean(axis=0)
    # print(averageCurve_app)
    averageCurve.append(averageCurve_app)
    
    
    
# Clustered averaged curves

for i in range(1,8):
    plt.figure(2)
    plt.plot(t,averageCurve[i])
    plt.xlabel('time(mins)')
    plt.ylabel('$C_t$ (mM)')
    plt.title("Averaged tissue concentration curve clusters")

averageCurve = np.asarray(averageCurve)


    
AIF_new_plots = []
ff_param_list = []
VEF_sub_list = []
VEF_ratio = []
Ranges = []
New_AIFs = []
maxpointlist =[]


''' 

Blind estimation clustering technique 

'''

H = 10
subset = 500
Minus = np.zeros((H,subset,len(AIF)))
rangeMinus = np.zeros((H,subset))

for h in range(H):
        
    
    D = random.choices(ConcTiss, k=subset) 
    for i in range(subset):
        
        # Initial AIF values
        A = [0.25, 0.475, 0.1] 
        
        
        # Random number generator
        randnum = randint(0, 7)    # Pick a random number between 1 and 100.
        
        # Randomly generated noisy curves from the averaged clusters
        ys2 = averageCurve[randnum,:]
        
    
        # This function assesses the sum of differences using the initial AIF
        def residual(AIF_initial_values):   
            return Extended_Tofts_Integral(t, AIF, *AIF_initial_values, uniform_sampling=True) - ys2
        
        # This function assesses the sum of differences using the estimated AIF + the functional form parameters
        def residual2(P, t, noise_Ct):
            return Extended_Tofts_Integral2(t,*P, uniform_sampling=True) - ys2 
                
        # Performing least squares with the initial values 
        AIF_res1 = least_squares(residual, A)
    
        
        # Turning aray into a list
        B = np.ndarray.tolist(AIF_res1.x)
            
        # Starting values for the AIF estimation
        Param = [A0,A1,A2,A3,Delta0,a0,a1,tau0,tau1,tau2,T, B[0], B[1], B[2]]
        
        # Performing least squares with the estimated values 
        try:
            
            C, pcov2 = optimize.leastsq(residual2, Param, args=(t, ys2), maxfev=5000)
        
        except OverflowError:   
            continue
        
        C_end = [C[9], C[10], C[11]]
        
        CC = C[0:11]
        
        # New AIF  
        AIF_new = AIF_Fun_SP(*CC, t)
        New_AIFs.append(AIF_new)
        plt.figure()    
        plt.plot(t, AIF)
        plt.plot(t, AIF_new)
        plt.xlim([0,2])
        plt.legend(['AIF before', 'AIF After'], loc="upper right")
        plt.show()
        
        for j in range(1,len(AIF)):
            
            L = i
            Minus[h,i,j] = ((New_AIFs[L][j]-AIF[j])**2)
           
    
        rangeMinus[h,i] = max(Minus[h,i]) - min(Minus[h,i])    
        
        # Collecting the data for the converged plots
                
        if rangeMinus[h,i] < 0.1:
                # Plot data
                AIF_new_plots.append(AIF_new)
                
                # Parameter data
                ff_param_list.append(C)
        
       
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   




        