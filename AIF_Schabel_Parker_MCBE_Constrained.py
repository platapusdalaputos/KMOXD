#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 11:41:23 2022

@author: Yohn Taylor
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import math
import random
from sklearn.cluster import KMeans
from lmfit import minimize, Parameters
from concentrationFunc_AIF import AIF_Fun_SP2, Extended_Tofts_Integral


''' Parameters associated with the Tofts model'''

# # AIF parameters

Ktrans = 0.1
Ve = 0.3
Vp = 0.04
Kep = Ktrans/Ve
Hct = 0.42

# Functional form Schabel and Parker
A1 = 6 
A2 = 1.1208
A3 = 0.3024 
A4 = 0.7164

# % Array of variables for A
A = [ A1, A2, A3, A4]

Delta1 = 1; 
Delta2 = Delta1+0.2227 
Delta3 = Delta1+0.3024 
Delta4 = Delta3

Delta = [Delta1, Delta2, Delta3, Delta4]

a1 = 2.92
a2 = a1
a3 = a1 
a4 = a1

# Array of variables for a
a = [a1, a2, a3, a4]

tau1 = 0.0442
tau2 = 0.1430
tau3 = tau2
tau4 = tau2

# Array of variables for tau
tau = [tau1, tau2, tau3, tau4];

T = 7.8940
N = 20
M = 1000
t = np.linspace(0,N,M)


Param = np.array([A1,A2,A3,A4,Delta1,Delta2,Delta3,a1,tau1,tau2,T])

AIF = AIF_Fun_SP2(*Param, t)

plt.figure(1)
plt.plot(t, AIF)
plt.ylabel("Cp")


# Uniform random distribution of the kinetic parameters
Num = 5000

Ktrans = np.random.uniform(0,0.5,Num)
Ve = np.random.uniform(0.15,1,Num)
Vp = np.random.uniform(0,0.2,Num)


# Adding guassian noise (mM) with mean = 0, and Std = 0.08e-3 M
noise = np.random.normal(0.04,0.08,M) 


# # Plotting the generated random uniform distributions

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


''' Generating the clusters'''

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

a = 0

for i in range(1,len(Ten_sets_of_four)):
    Ten_sets_of_four[i,:] = ConcTiss[a,:]
    a = i*10
    
plt.figure(4)
plt.plot(t, Ten_sets_of_four[2,:],'o')


# Cluster the data from the randomly generated tissue concentration curves

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
        plt.xlim([0,5])
        plt.ylabel(' $C_t$ (mM)')
        plt.title("Multiple tissue concentration curve clusters")
        a = np.asarray(clustered_average)       
    averageCurve_app = a.mean(axis=0)
    # print(averageCurve_app)
    averageCurve.append(averageCurve_app)
    
    
    
# Clustered averaged curves

for i in range(1,8):
    plt.figure(2)
    plt.plot(t,averageCurve[i],'-')
    plt.xlabel('time(mins)')
    plt.ylabel('$C_t$ (mM)')
    plt.title("Averaged tissue concentration curve clusters")
    plt.xlim([0,5])
    # plt.grid()

averageCurve = np.asarray(averageCurve)


def residual(params, t, ys2):
    
    Ktrans = params['Ktrans']
    Ve = params['Ve']
    Vp = params['Vp']
    
    Initial_Values1 = [Ktrans, Ve, Vp]
        
    Diff = Extended_Tofts_Integral(t, AIF,*Initial_Values1, uniform_sampling=True) - ys2
    
    return Diff


def residual2(params2, t, ys2):
    
    Ktrans = params2['Ktrans']
    Ve = params2['Ve']
    Vp = params2['Vp']
    
    A1 = params2['A1']
    A2 = params2['A2']
    A3 = params2['A3']
    A4 = params2['A4']
    Delta1 = params2['Delta1']
    Delta2 = params2['Delta2']
    Delta3 = params2['Delta3']
    a1 = params2['a1']
    tau1 = params2['tau1']
    tau2 = params2['tau2']
    T = params2['T']
    
    
    Initial_Values2 = [A1,A2,A3,A4,Delta1,Delta2,Delta3,a1,tau1,tau2,T,Ktrans, Ve, Vp]        
    Diff2 = Extended_Tofts_Integral2_SP(t,*Initial_Values2, uniform_sampling=True) - ys2
    return Diff2

    
AIF_new_plots = []
ff_param_list = []
New_AIFs = []


''' 

Monte Carlo Blind Estimation (MCBE) technique 

'''



Q = 256

Curves = [2,3,4,5,6]

for g in range(len(Curves)):
    
    for h in range(Q):
        
        D = random.choices(ConcTiss, k=Curves[g]) 
        
        AIF_VALUES = []
        
        for i in range(Curves[g]):
            
                            
            # Initial paramerters for the initial VEF measurements   
            params = Parameters()
            params.add('Ktrans', value=0.25, min = 0, max = math.inf)
            params.add('Ve', value=0.475, min = 0, max = math.inf)
            params.add('Vp', value=0.1, min = 0, max = math.inf)
                        
            # Random.choices allows replacement
            ys2 = D[i]
                   
            # Extracting the parameter values for the new Total concentration curve after leastsq analysis 
            Leastsq_Value1 = minimize(residual, params, args=(t, ys2))
            K = Leastsq_Value1.params         
            
            # New parameter set for the functional form (using Schabel and parker parameters)
            K.add('A1', value = 1)
            K.add('A2', value = 1, min = 0, max = 1)
            K.add('A3', value = 1, min = 0, max = 1)
            K.add('A4', value = 1, min = 0, max = 1)
            K.add('Delta1', value = 1, min = -0.5, max = 0.5)
            K.add('Delta2', value = 1, min = -0.5, max = 1.5)
            K.add('Delta3', value = 1, min = -0.5, max = 2.5)
            K.add('a1', value = 1, min = 1, max = math.inf)
            K.add('tau1', value = 1, min = 0, max = math.inf)
            K.add('tau2', value = 1, min = 0, max = math.inf)
            K.add('T', value = 1, min = 0, max = math.inf)
    
            # Performing least squares on the functional form and the leastsq_Values1
            Leastsq_Value2 = minimize(residual2, K, args=(t, ys2))
            
            
            # Converting from a dictionary to a numpy array
            K2 = Leastsq_Value2.params
            AIF_Param_dict = K2.valuesdict()
            AIF_param_values = np.array(list(AIF_Param_dict.values()))   
            AIF_VALUES.append(AIF_param_values)        
            
                        
            AIF_final_params = AIF_param_values[0:11]
            ff_param_list.append(AIF_final_params)
            AIF_new = AIF_Fun_SP2(*AIF_final_params, t)
    
            plt.figure()    
            plt.plot(t, AIF)
            plt.plot(t, AIF_new)
            plt.xlim([0,2])
            plt.ylabel("Conc (mM)")
            plt.xlabel("time (s)")
            plt.legend(['AIF before', 'AIF After'], loc="upper right")
            plt.show()
            
            
            
            
            
            # Performing least squares with the estimated values 
            # try:
                
            #     C, pcov2 = optimize.leastsq(residual2, Param, args=(t, ys2), maxfev=5000)
            
            # except OverflowError:   
            #     continue
                
                
                # # C, pcov2 = optimize.least_squares(residual2, Param, args=(t, ys2), method = 'lm')
                # C_end = [C[10], C[11], C[12]]
                # # print(C_end)
                
                # CC = C[0:12]
                
                # # New AIF  
                # AIF_new = AIF_Fun_SP(*CC, t)
                
                # # plotting the new AIF
                # New_AIFs.append(AIF_new)
                # # plt.figure()    
                # # plt.plot(t, AIF)
                # # plt.plot(t, AIF_new)
                # # plt.xlim([0,2])
                # # plt.legend(['AIF before', 'AIF After'], loc="upper right")
                # # plt.show()
                
                # for j in range(1,len(AIF)):
                    
                #     L = i
                #     Minus[h,i,j] = ((New_AIFs[L][j]-AIF[j])**2)
                   
            
                # rangeMinus[h,i] = max(Minus[h,i]) - min(Minus[h,i])    
                
                # # Collecting the data for the converged plots
                       
                # if rangeMinus[h,i] < 0.1:
                #         AIF_new_plots.append(AIF_new)
                #         ff_param_list.append(C)
                
               
    
   
    
   
    
   
    
   
    
   
    
   
    
   
