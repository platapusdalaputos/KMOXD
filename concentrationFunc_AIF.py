
''' This file contains the different function used in Blind estimation of the AIF '''

import numpy as np
from scipy.special import gamma, gammainc
from scipy import integrate 
import matplotlib.pyplot as plt



# Gamma variate curve for the schabel and parker plot

def gammaVariate(t, a, tau):

    if t<0:
        G = 0    
    elif t>=0:
    
        p1 = (np.exp(1)/(tau*(a-1)))**(a-1)
        p2 = t**(a-1)
        p3 = np.exp(-t/tau)
    
        G = p1*p2*p3
         
    return G 

# Sigmoid Curve for the Schabel and Parker plot

def SigmoidCurve(t,a,tau,T):

    if t<0:
        Sig = 0    
    elif t>=0:
        
        p1 = (T/((T-tau)*(gamma(a))));
        p11 = (t/T)**(-tau/(T-tau));
        p2 = np.exp(-t/T);
        # p3 = ((1/tau)-(1/T))*t;
        # p4 = sc.gammainc(a,(p3));
        p4 = gammainc(a,(((1/tau)-(1/T))*t));
        
        Sig = p1*p11*p2*p4;
         
    return Sig

# AIF Schabel and Parker paramweters

t = np.linspace(0,5,1000)
    
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
tau3 = 0.14

T = 9.6319

# Schabel and Parker Functional form 
def AIF_Fun_SP(A0,A1,A2,A3,Delta0,a0,a1,tau0,tau1,tau2,T,t):

    
    # Variables for the Schabel and Parker AIF
    
    
    # Array of variables for A
    A = [A0, A1, A2, A3]
    
    Delta1 = Delta0
    Delta2 = Delta0
    Delta3 = Delta0
    
    # Array of variables for Delta
    Delta = [Delta0, Delta1, Delta2, Delta3]
    
    
    # Array of variables for a
    a2 = a0
    a3 = a0
    a = [a0, a1, a2, a3]
    
    
    
    # Array of variables for tau
    tau3 = tau0
    tau = [tau0, tau1, tau2, tau3];
    
    # T = 9.6319
    # N = 20
    # t = np.linspace(0,N,1000)
    
    Cp = []
    A0Sig =[]
    AnGamma = []
    # aaa = np.zeros((len(t),3))
    
    for i in range(len(t)):
        Sig = SigmoidCurve(t[i]-Delta0,a0,tau0,T)
        A0S = A0*Sig;
        A0Sig.append(A0S)
    
        # AnG stands for An * G
        AnG = []
        for j in range(1,4):
            G = gammaVariate(t[i]-Delta[j],a[j],tau[j])
            AnG.append(A[j]*G) 
        
        AnGi = sum(AnG)
        AnGamma.append(AnGi)
        Cp.append(A0S + AnGi)
    Cblood = np.array(Cp)
    
    return Cblood
# plt.plot(t,AIF_Fun_SP(A0,A1,A2,A3,Delta0,a0,a1,tau0,tau1,tau2,T,t))

def AIF_Fun_SP2(A1,A2,A3,A4,Delta1,Delta2,Delta3,a1,tau1,tau2,T,t):

    
    # Variables for the Schabel and Parker AIF

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
    t = np.linspace(0,N,1000)
    
    Cp = []
    A0Sig =[]
    AnGamma = []
    # aaa = np.zeros((len(t),3))
    
    for i in range(len(t)):
        Sig = SigmoidCurve(t[i]-Delta0,a0,tau0,T)
        A0S = A4*Sig;
        A0Sig.append(A0S)
    
        # AnG stands for An * G
        AnG = []
        for j in range(0,3):
            G = gammaVariate(t[i]-Delta[j],a[j],tau[j])
            AnG.append(A[j]*G) 
        
        AnGi = sum(AnG)
        AnGamma.append(AnGi)
        Cp.append(A0S + AnGi)
    Cblood = np.array(Cp)
    
    return Cblood

# Geoff Parker Functional form 
def AIF_fun(Param,Hct,Vp,Kep,t):
    n1 = []
    n2 = []
    Cb = []
    p1 = []
    k1 = []
    Cp = []
    Ct = []
    
    ''' Plasma concentration (AIF) '''
    
    for i in range(len(t)):  
        
        n1a = (Param[0]/(Param[2]*np.sqrt(2*np.pi)))*np.exp((-(t[i]-Param[4])**2)/(2*(Param[2]**2)))
        n1.append(n1a)
        
        n2a = (Param[1]/(Param[3]*np.sqrt(2*np.pi)))*np.exp((-(t[i]-Param[5])**2)/(2*(Param[3]**2)))
        n2.append(n2a)
        
        Cba = n1[i] + n2[i] + (Param[7]*np.exp(-Param[6]*t[i]))/(((1+np.exp(-Param[8]*(t[i]-Param[9])))))
        Cb.append(Cba)
        
        Cpa = Cb[i]*(1-Hct);
        Cp.append(Cpa)
        
        p1a = Vp*Cp[i];
        p1.append(p1a)
        
        k1a = np.exp(-Kep*t[i]);
        k1.append(k1a)
        
    
    Cp = np.array(Cp)
    Cb = np.array(Cb)
    return Cb


# Tofts intergral using the initial kinetic parameters
def Extended_Tofts_Integral(t, Cp, Kt, ve, vp, uniform_sampling=True):
# def Extended_Tofts_Integral(t, Cp, Kt=0.1, ve=0.2, vp=0.1, uniform_sampling=True):
    nt = len(t)
    Ct = np.zeros(nt)
    for k in range(nt):
        tmp = vp*Cp[:k+1] + integrate.cumtrapz(np.exp(-Kt*(t[k]-t[:k+1])/ve)*Cp[:k+1],t[:k+1], initial=0.0)
        # print(tmp)
        Ct[k] = tmp[-1]
        # print(Ct[k])
    return Ct


# Extended Tofts integral using the Schabel and Parker parameters
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

# Extended Tofts integral using the Schabel and Parker parameters part 2
def Extended_Tofts_Integral2_SP(t, A1,A2,A3,A4,Delta1,Delta2,Delta3,a1,tau1,tau2,T, Kt, ve, vp, uniform_sampling=True):
    Cp = AIF_Fun_SP2(A1,A2,A3,A4,Delta1,Delta2,Delta3,a1,tau1,tau2,T, t)       
    nt = len(t)
    Ct = np.zeros(nt)
    for k in range(nt):
        tmp = vp*Cp[:k+1] + integrate.cumtrapz(np.exp(-Kt*(t[k]-t[:k+1])/ve)*Cp[:k+1],t[:k+1], initial=0.0)
        # print(tmp)
        Ct[k] = tmp[-1]
        # print(Ct[k])
    return Ct



