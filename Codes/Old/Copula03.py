import numpy as np
from numpy import exp, log, random, array, arange, zeros, sqrt, ones
from numpy import tile, argmin
from scipy.integrate import trapz, cumtrapz
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
    
#rho0 = np.corrcoef(Z,X)[0,1]
#%%
k = 8
dx = 1e-2   
dz = dx#*sdz/sdx 
    
sdx1 = sqrt(2)
x0 = (1-k*sdx).round(2)
xGrid = arange(x0,1+k*sdx,dx)
fx = norm.pdf(xGrid,sdx,sdx1)
Fx = norm.cdf(xGrid,sdx,sdx1)
Fz = norm.cdf(Z[0],0,sqrt(5))
copula = c(Fx,Fz)
pn = copula*fx
x1Grid = tile(xGrid,(1,1))
#%%
for t in range(1,T-1):
    mux1 = fmu(X[t]).round(2)
    x0 = (mux1-k*sdx).round(2)
    xGrid = x1Grid[0]
    x1Grid = arange(x0,2*mux1-x0,dx)
    nx0 = len(xGrid)
    nx1 = len(x1Grid)
    x1Grid = np.tile(x1Grid,(nx0,1))
    x0Grid = np.tile(fmu(xGrid),(nx1,1))
    #xGrid2 = np.arange(2-k,2+k,dx)
    #xGrid2 = np.tile(xGrid2,(nx,1)).T
    integrand = norm.pdf(x1Grid.T,x0Grid,sdx)*pn
    fx = trapz(integrand,dx=dx,axis=1)
    Fx = cumtrapz(fx,dx=dx,initial=None)
    Fx = np.append(Fx,1-0.5*(1-Fx[-1]))
    #%%
    muz0 = X[t].round(2)
    z0 = (muz0-k*sdz).round(2)
    z0Grid = arange(z0,2*muz0-z0,dz)
    nz0 = len(z0Grid)
    z0Grid = tile(z0Grid,(nx0,1))
    x0Grid = tile(xGrid,(nz0,1))
    integrand = norm.pdf(z0Grid.T,x0Grid,sdz)*pn
    zIdx = np.argmin(abs(Z[t]-z0Grid[0]))
    fz = trapz(integrand,dx=dz,axis=1)
    Fz = cumtrapz(fz[:zIdx+1],dx=dz)[-2:].mean()
    pn = c(Fx,Fz)*fx
    
    #%%
    plt.figure()
    title = f'$p{t+1}$'
    plt.plot(x1Grid[0],pn)
    plt.axvline(x=X[t+1],color='black')
    plt.title(title)
#%%
#plt.plot(z0Grid[0],fz)
#plt.axvline(x=Z[t],color='black')
#plt.title('f(z)')