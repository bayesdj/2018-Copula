from numpy import exp, log, random, array, arange, zeros, sqrt, ones
from numpy import tile, argmin, linspace, stack, outer
import numpy as np
from scipy.integrate import trapz, cumtrapz, romb
from scipy.special import gamma
import scipy.stats as stats
from scipy.stats import norm, cauchy
import matplotlib.pyplot as plt
#%%
def fmu(x):
    return x+1

def c(u,v):
    s = norm.ppf(u)
    t = norm.ppf(v)
    return rho2inv*exp(-0.5*(rho2*s*s-2*rho*s*t+rho2*t*t)*rho2inv*rho2inv)

def graphDensity(t,xGrid,density,pnmc):
    title = rf'p{t} $\rho$={rho}'
    n = len(pnmc)
    plt.figure()
    plt.plot(xGrid,density)
    plt.hist(pnmc,bins=200,density=True)
    plt.axvline(x=X[t],color='black')
    plt.legend([rf'Kalman','copula',f'x{t}',f'mc n={n}'])
    plt.title(title)
    
def graphDensity2(dt,t,xGrid,density,pnmc):
    if t%dt == 0:
        k = int(t/dt)
        i,j = divmod(k,2)
        h = axes[i-1,j]
        title = rf'p{t} $\rho$={rho}'
        n = len(pnmc)
        h.plot(xGrid,density)
        h.hist(pnmc,bins=200,density=True)
        h.axvline(x=X[t],color='black')
        h.legend([rf'Kalman','copula',f'x{t}',f'mc n={n}'])
        h.set_title(title)
#%%
rho = 0.5
rho2 = rho*rho
rho2inv = 1/sqrt(1-rho2)
    
T = 31
varx = 1
sdx = sqrt(varx)
varz = 4
sdz = sqrt(varz)

#%%
X = zeros(T)
Z = zeros(T)
W = Z.copy()
Mu = Z.copy()

x = random.normal(0,sdx)
X[0] = x
z = random.normal(x,sdz)
Z[0] = z
w = 0; mu = 0
W[0] = w; Mu[0] = mu

for t in range(1,T):    
    x = fmu(x)+random.normal(0,sdx)
    z = x+random.normal(0,sdz)
    varx1 = varx+w
    w = varz*varx1/(varz+varx1)
    mu = w*(z/varz+(1+mu)/varx1)
    X[t] = x
    Z[t] = z
    W[t] = w
    Mu[t] = mu
    
#%%
k = 10 
nx = 2**10+1
Fmin = 1e-16
Fmax = 1-Fmin
    
sdx1 = sqrt(2)
xGrid,dx0 = linspace(X[1]-k*sdx,X[1]+k*sdx,nx,retstep=True)
fx = norm.pdf(xGrid,1,sdx1)
Fx = norm.cdf(xGrid,1,sdx1)
Fz = romb(norm.cdf(Z[0],xGrid,sdz)*fx,dx=dx0)
cop = c(Fx,Fz)
pn = cop*fx
pnTrue = norm.pdf(xGrid,fmu(Mu[0]),sqrt(W[0]+varx))
#plt.plot(xGrid,cop)
#graphDensity(1,xGrid,stack([pn,pnTrue]).T)
#%%
mcN = int(3e4)
onex = ones(mcN)
x0 = norm.rvs(loc=1,scale=sdx1,size=mcN)
Fz = norm.cdf(Z[0],loc=x0,scale=sdz).mean()
Fx = norm.cdf(x0,1,scale=sdx1)
copmc = c(Fx,Fz)

#mcN = int(1e4)
u = random.uniform(size=mcN)
idx = random.choice(len(x0),size=mcN)
M = copmc.max()
pnTry = x0[idx]
pnmc = pnTry[(M*u)<copmc[idx]]
#%%
h = plt.hist(pnmc,density=True,bins=200)
h = plt.plot(xGrid,pn)

#%%
#mcNx = mcN #int(3e4)
fig,axes = plt.subplots(3,2,sharex=True,sharey=True,figsize=(10,10))

#oney = ones(mcNy)
for t in range(1,T-1):
    exp_xt = romb(xGrid*pn,dx=dx0)
    exp_mc = pnmc.mean()
    muxt = fmu(exp_mc)    
    end0 = muxt-k*sdx; end1 = muxt+k*sdx
    x1Grid,dx1 = linspace(end0,end1,nx,retstep=True)
    x1Grid = tile(x1Grid,(nx,1))
    x0Grid = tile(fmu(xGrid),(nx,1))
    integrand = norm.pdf(x1Grid.T-x0Grid,loc=0,scale=sdx)*pn
    fx = romb(integrand,dx=dx0,axis=1) # numerical integration
    Fx = cumtrapz(fx,dx=dx1,initial=Fmin) # numerical integration
    Fx[Fx>=Fmax] = Fmax
    
    integrand = norm.cdf(Z[t],xGrid,sdz)*pn
    Fz = romb(integrand,dx=dx0) # numerical integration
    cop = c(Fx,Fz)
    pn = cop*fx
    xGrid = x1Grid[0]
    dx0 = dx1
    pnTrue = norm.pdf(xGrid,fmu(Mu[t]),sqrt(W[t]+varx))
    
    x1 = random.uniform(end0,end1,size=mcN)
    e = outer(x1,ones(len(pnmc)))-outer(fmu(pnmc),onex).T
    fxy = norm.pdf(e,scale=sdx).mean(axis=1)
    u = random.uniform(high=fxy.max(),size=mcN)
    idx = u <= fxy
    fxmc = x1[idx]
#    idxfx = fxmc.argsort()
#    n = len(fxmc)
    Fx = norm.cdf(e[idx],scale=sdx).mean(axis=1)
    Fz = norm.cdf(Z[t]-pnmc,scale=sdz).mean()
    copmc = c(Fx,Fz)
    M = copmc.max()
    u = random.uniform(size=mcN)
    idx = random.choice(len(fxmc),size=mcN)
    pnTry = fxmc[idx]
    pnmc = pnTry[(M*u)<copmc[idx]]
    
#    h = plt.hist(fxmc,bins=200,density=True)
#    plt.plot(x1Grid[0],fx)
#    plt.figure()
#    h = plt.hist(pnmc,bins=200,density=True)
#    plt.plot(x1Grid[0],pn)
    
    graphDensity2(5,t+1,xGrid,stack([pnTrue,pn]).T,pnmc)
