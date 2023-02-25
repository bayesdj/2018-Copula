import numpy as np
from numpy import exp, log, random, array, arange, zeros, sqrt, ones
from numpy import tile, argmin, linspace
from scipy.integrate import trapz, cumtrapz, romb
from scipy.stats import norm
import matplotlib.pyplot as plt
#%%
T = 6
varx = 1
sdx = sqrt(varx)
varz = 4
sdz = sqrt(varz)

X = zeros(T)
Z = zeros(T)
W = Z.copy()
Mu = Z.copy()

random.seed(46)
x = random.normal(0,sdx)
X[0] = x
z = random.normal(x,sdz)
Z[0] = z
w = 0; mu = 0
W[0] = w; Mu[0] = mu

rho = 0.5
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
    varx1 = varx+w
    w = varz*varx1/(varz+varx1)
    mu = w*(z/varz+(1+mu)/varx1)
    X[t] = x
    Z[t] = z
    W[t] = w
    Mu[t] = mu
    
#%%
k = 8
dx = 1e-2   
    
sdx1 = sqrt(2)
x0 = (1-k*sdx).round(2)
xGrid = arange(x0,1+k*sdx,dx)
fx = norm.pdf(xGrid,sdx,sdx1)
Fx = norm.cdf(xGrid,sdx,sdx1)
Fz = norm.cdf(Z[0],0,sqrt(5))
copula = c(Fx,Fz)
pn = copula*fx
#x1Grid = tile(xGrid,(1,1))
#%%
for t in range(1,T-2):
    muxt_real = fmu(X[t])
    muxt = muxt_real.round(2)
    x0 = (muxt-k*sdx).round(2)
    x1Grid = arange(x0,2*muxt-x0,dx)
    nx0 = len(xGrid)
    nx1 = len(x1Grid)
    x1Grid = np.tile(x1Grid,(nx0,1))
    x0Grid = np.tile(fmu(xGrid),(nx1,1))
    integrand = norm.pdf(x1Grid.T,x0Grid,sdx)*pn
    fx = romb(integrand,dx=dx,axis=1)
    Fx = cumtrapz(fx,dx=dx,initial=None)
    Fx = min(1-1e-15,np.append(Fx,1-0.5*(1-Fx[-1])))
    
    integrand = norm.cdf(Z[t],xGrid,sdz)*pn
    Fz = trapz(integrand,dx=dx)
    pn = c(Fx,Fz)*fx
    xGrid = x1Grid[0]
    pnTrue = norm.pdf(xGrid,fmu(Mu[t]),sqrt(W[t]+varx))
    
    plt.figure()
    title = f'$p{t+1}$'
    plt.plot(xGrid,array([pn,pnTrue]).T)
    plt.axvline(x=X[t+1],color='black')
    plt.title(title)
    plt.legend([rf'copula $\rho$={rho}','Kalman'])
#%%
#plt.plot(z0Grid[0],fz)
#plt.axvline(x=Z[t],color='black')
#plt.title('f(z)')