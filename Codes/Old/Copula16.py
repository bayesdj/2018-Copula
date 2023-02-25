from numpy import exp, log, random, array, arange, zeros, sqrt, ones
from numpy import tile, argmin, linspace, stack, outer, empty
import numpy as np
from scipy.integrate import trapz, cumtrapz, romb
from scipy.optimize import newton, bisect
from scipy.stats import norm
import matplotlib.pyplot as plt
from itertools import repeat
import time
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
        i,j = divmod(k-1,2)
        h = axes[i,j]
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
    
T = 6
varx = 1
sdx = sqrt(varx)
varz = 4
sdz = sqrt(varz)
fxyMax = norm.pdf(0,loc=0,scale=sdx)
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
MCN = int(1e4)
mcN = int(1e4)
IDX = arange(MCN)
onex = ones(MCN)
x0 = norm.rvs(loc=1,scale=sdx1,size=MCN)
Fz = norm.cdf(Z[0],loc=x0,scale=sdz).mean()
Fx = norm.cdf(x0,loc=1,scale=sdx1)
copmc = c(Fx,Fz)

#u = random.uniform(size=mcN)
#idx = random.choice(len(x0),size=mcN)
M = c(norm.cdf(norm.ppf(Fz)/rho),Fz)
prob = copmc/M
prob /= prob.sum()
#pnTry = x0[idx]
#pnmc = pnTry[(M*u)<copmc[idx]]
pnmc = random.choice(x0,size=mcN,p=prob)

h = plt.hist(pnmc,density=True,bins=200)
h = plt.plot(xGrid,pn)
#%%

def getFx(x,mu,u):
    dx = x - mu
    cdf = norm.cdf(dx,scale=sdx).mean()
    return cdf-u

def getfx(x,mu,u):
    dx = x-mu
    pdf = norm.pdf(dx,scale=sdx).mean()
    return pdf

def mapNewton(x0,mu,u):
    return newton(getFx,x0=x0,fprime=getfx,args=(mu,u))    
    
tol = 1e-6
n = 10000
#%%
def buildRev(fxmc,Fx,pnmc0,tol,n,Fz):
    panel = np.stack((fxmc,Fx),axis=0).T[fxmc.argsort()]
    U = random.uniform(size=n)
    U1 = norm.ppf(U,loc=rho*norm.ppf(Fz),scale=sqrt(1-rho2))
    U1 = norm.cdf(U1)

    mu = fmu(pnmc0) # for newton
    idxU = np.searchsorted(panel[:,1],U1)
    xHi = panel[-1,0] + 0.01*abs(panel[-1,0])
    fxmcSorted = np.append(panel[:,0],xHi)
    x0 = fxmcSorted[idxU]
    t0 = time.time()
#    %%timeit
    g = map(mapNewton,x0,repeat(mu,n),U1)
    X = np.fromiter(g,dtype=np.float)
    #X = empty(n)
#    for i in range(n):
#        X[i] = newton(getFx,x0=x0[i],fprime=getfx,args=(mu,U1[i]))     
    t1 = time.time()
    print(t1-t0)
    return X
        
#%%
    
fig,axes = plt.subplots(3,2,sharex=False,sharey=True,figsize=(10,10))
accept = np.empty(T)

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
    
    x1 = random.uniform(end0,end1,size=MCN)
    e = outer(x1,ones(len(pnmc)))-outer(fmu(pnmc),onex).T
    fxy = norm.pdf(e,scale=sdx).mean(axis=1)
#    prob = fxy/fxyMax
#    idx = random.choice(IDX,size=mcN,p=prob/prob.sum())
#    fxmc = x1[idx]
    u = random.uniform(high=fxyMax,size=mcN)
    idx = u <= fxy
    fxmc = x1[idx]

    Fx = norm.cdf(e[idx],scale=sdx).mean(axis=1)
    Fz = norm.cdf(Z[t]-pnmc,scale=sdz).mean()
    copmc = c(Fx,Fz)
    M = c(norm.cdf(norm.ppf(Fz)/rho),Fz) # maximum of the copula distribution
    u = random.uniform(size=mcN)
    idx = random.choice(len(fxmc),size=mcN)
    pnTry = fxmc[idx]
    pnmc0 = pnmc
    pnmc = pnTry[(M*u)<copmc[idx]]
    accept[t] = len(pnmc)/len(pnTry)
#    prob = copmc/M
#    pnmc = random.choice(fxmc,size=mcN,p=prob/prob.sum())
    
#    graphDensity2(5,t+1,xGrid,stack([pnTrue,pn]).T,pnmc)
    print(t)
