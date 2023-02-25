import numpy as np
from numpy import exp, log, random, array, arange, outer, zeros, sqrt, ones
from scipy.integrate import trapz, cumtrapz
from numpy import linalg as la
from scipy.stats import norm
import matplotlib.pyplot as plt
#%%
T = 50
varx = 1
sdx = sqrt(varx)
varz = 4
sdz = sqrt(varz)

X = zeros(T)
Z = zeros(T)

#random.seed(6)
x = random.normal(0,sdx)
X[0] = x
z = random.normal(x,sdz)
Z[0] = z

rho = 0.3
rho2 = rho*rho
rho2inv = 1/(1-rho2)

def fmu(x):
    return x+1

def c(u,v):
    s = norm.ppf(u)
    t = norm.ppf(v)
    return sqrt(rho2inv)*exp(-0.5*(rho2*s*s-2*rho*s*t+rho2*t*t)*rho2inv)
#%%

for t in range(1,T):    
    x = random.normal(fmu(x),sdx)
    z = random.normal(x,sdz)
    X[t] = x
    Z[t] = z
    
rho0 = np.corrcoef(Z,X)[0,1]

sdx1 = sqrt(2)
k = 6
dx = 1e-2
#xGrid = np.linspace(1-k*sdx1,1+k*sdx1,2000)
xGrid = arange(1-k,1+k,dx)
nx = len(xGrid)
fx = norm.pdf(xGrid,1,sdx1)
Fx = norm.cdf(xGrid,1,sdx1)
Fz = norm.cdf(Z[0],0,sqrt(5))

pn1 = c(Fx,Fz)*fx
xGrid1 = np.tile(xGrid,(nx,1))
xGrid2 = np.arange(2-k,2+k,dx)
xGrid2 = np.tile(xGrid2,(nx,1)).T
intgrad = norm.pdf(xGrid2,xGrid1+1,1)*pn1
#intgrad = intgrad*pn1
fx2 = trapz(intgrad,dx=dx,axis=0)
Fx2 = cumtrapz(fx2,x=xGrid2[:,0],initial=None)

#%%
plt.plot(xGrid,pn1)
plt.axvline(x=X[1],color='black')