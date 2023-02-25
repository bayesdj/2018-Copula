from numpy import exp, log, random, array, arange, zeros, sqrt, ones
from numpy import tile, argmin, linspace, stack, outer
import numpy as np
import scipy.stats as stats
from scipy.stats import norm
from scipy.optimize import newton, root_scalar,brentq,brenth
import matplotlib.pyplot as plt
import timeit, time
from itertools import repeat
from numba import jit
# one step ahead prediction, x_{k+1} | z_k
#%%
def fmu(x):
    return x+1

def q(mu,N):
    return mu + rvx.rvs(size=N)

def graphDensity(t,xi,wi,nff,density):
    nff = round(nff,1)
    title = f'$p{t}$, '+r'$N_{eff}$ = '+f'{nff}'
    plt.figure()
    plt.plot(xGrid,density)
    plt.hist(xi,weights=wi,density=True,bins=300)
    plt.axvline(x=X[t],color='black')
    plt.title(title)
    
def graphDensity2(t,nff,densities,pnmc=None):
    nff = round(nff,1)
    title = f'p{t}, '+r'$N_{eff}$ = '+f'{nff}'
    leg = ['particle',f'x{t}']
    plt.figure()
    plt.plot(xGrid,densities)
    plt.axvline(x=X[t],color='black')
    if pnmc is not None:
        plt.hist(pnmc,density=True,bins=300)
        leg += ['mc']
    plt.legend(leg)
    plt.title(title)

def graphDensity3(t,nff,densities,pnmc):
    nff = round(nff,1)
    title = f'p{t}, '+r'$N_{eff}$ = '+f'{nff}'
    fig,axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(9,4))
    sub_titles = [rf'Gaussian Cop, $\rho$={rho}',
                  rf'Mixed Cop, $\rho$={rho2}']    
    for i in range(2):
        h = axes[i]
        h.plot(xGrid,densities)
        h.hist(pnmc[i],bins=250,density=True)
        h.axvline(x=X[t],color='black')
        h.set_title(sub_titles[i])
    fig.suptitle(title)
    #fig.show()
#%%
def CMix(x,u):
    return (alpha1)*x+alpha*rvu.cdf(norm._ppf(x))-u

def cMix(x,u):
    s = norm.ppf(x)
    return alpha*(exp(rvu.logpdf(s)-norm.logpdf(s))-1)

@jit(nopython=True)
def getFx(x,mu,u):
    cdf = rvx.cdf(x-mu).mean()
    return cdf-u

def getfx(x,mu,u):
    return rvx.pdf(x-mu).mean()

def root1(x0,x1,mu,u):    
    sol = root_scalar(getFx,args=(mu,u),method='brenth',
                       bracket=(x0,x1),options={'disp':False})  
    if sol.converged:
        return sol.root
    else:
        errorString = f'Did not converge on x0={x0},x1={x1},u={u}'
        raise Exception(errorString)

def getAlpha(t):
    #return (1/(2+t))**0.65
    return 1

#@jit(nopython=True)
def mixCop(U):
    U2 = np.empty(N) 
    x = 0
    for i in range(0,N,2):
        x = brentq(CMix,x,1,args=(U[i],),disp=False)
        U2[i] = x
    for i in range(1,N,2):
        x = brentq(CMix,U[i-1],U[i+1])
#        sol = root_scalar(CMix,args=(U[i],),method='newton',
#               x0=x,fprime=cMix,options={'disp':True})
#        if sol.converged:
#            x = sol.root
#            U2[i] = x
#        else:
#            errorString = f'Did not converge on x0={x},u={U[i]}'
#            raise Exception(errorString)
    return U2

def xFromU(U,Fx,xGrid,fpnmc):
    idxU = np.searchsorted(Fx,U)
    adj = 0
    m = 9999
    if idxU.min() == 0:
        v = xGrid[0]-abs(xGrid[0])*m
        #v = -np.inf
        xGrid = np.insert(xGrid,0,v)
        adj += 1
    if idxU.max() == ndx:
        v = xGrid[-1]+abs(xGrid[-1])*m
        #v = np.inf
        xGrid = np.append(xGrid,v)
    idxU = idxU+adj if adj > 0 else idxU
    X = np.empty(N)
    x0 = xGrid[idxU-1]
    x1 = xGrid[idxU]
    x = x0[0]
    for i in range(N):
        x = root1(x,x1[i],fpnmc,U[i])
        X[i] = x
    return X
#%%
varx = 1; sdx = sqrt(varx)
rvx = stats.gamma
gamma_a = 1

rvz = norm(scale=sqrt(2))
def tryx(v):
    return v._cdf(0)

g = tryx(rvx)

rho = 0.48
rt_rho = sqrt(1-rho*rho)
rho2 = 0.48
rt_rho2 = sqrt(1-rho2*rho2)

T = 5
N = int(3000)
k = 8
ndx = 2**11+1
Grid = linspace(-k*sdx,k*sdx,ndx,retstep=False)
#%%
X = zeros(T)
Z = zeros(T)
Thi = zeros(T)
Mu = zeros(T)

x = rvx.rvs()
X[0] = x
z = x + rvz.rvs()
Z[0] = z

for t in range(1,T):    
    x = fmu(x)+rvx.rvs()
    z = x+rvz.rvs()
    X[t] = x
    Z[t] = z
#%%
# particle filter 1
Ninv = 1/N*ones(N)
xi = q(0,N)
t = 0
alpha = getAlpha(t)
alpha1 = 1-alpha
wi = rvz.pdf(Z[t]-xi)
wi /= wi.sum()
nff = 1/(wi@wi)
muXt = fmu(xi@wi)
xGrid = muXt + Grid
x1density = wi@rvx.pdf(tile(xGrid,(N,1))-tile(fmu(xi),(ndx,1)).T)
# kalman = norm.pdf(xGrid,fmu(Mu[t]),Thi[t])
#%%
pnmc = xi.copy()
fpnmc = fmu(pnmc)
Fx = rvx.cdf(tile(xGrid,(N,1))-tile(fpnmc,(ndx,1)).T).mean(0)
Fz = rvz.cdf(Z[t]-pnmc).mean()
rvu = norm(loc=rho*norm.ppf(Fz),scale=rt_rho2)
#%%
N = 1000
U = random.uniform(size=N); U.sort()
#%%
def mixCop(U):
    U2 = np.empty(N) 
    x = 0
    for i in range(0,N,2):
#        x = brentq(CMix,x,1,args=(U[i],),disp=False)
        x = root_scalar(CMix,args=(U[i],),x0=x,x1=1,method='brentq',bracket=(x,1),options={'disp':False}).root
        U2[i] = x
    if N%2 == 0:
        U2[-1] = brentq(CMix,x,1,args=(U[-1],),disp=False)
    for i in range(1,N-1,2):
        #U2[i] = brentq(CMix,U2[i-1],U2[i+1],args=(U[i],),disp=False)
        a = U2[i-1]; b = U2[i+1]-a
        U2[i] = root_scalar(CMix,args=(U[i],),x0=a+0.35*b,x1=a+0.67*b,method='secant',bracket=(U2[i-1],U2[i+1]),
          options={'disp':False}).root
    return U2
#%% 
%timeit U2 = mixCop(U)
#%%
def xFromU(U,Fx,xGrid,fpnmc):
    idxU = np.searchsorted(Fx,U)
    adj = 0
    m = 9999
    if idxU.min() == 0:
        v = xGrid[0]-abs(xGrid[0])*m
        #v = -np.inf
        xGrid = np.insert(xGrid,0,v)
        adj += 1
    if idxU.max() == ndx:
        v = xGrid[-1]+abs(xGrid[-1])*m
        #v = np.inf
        xGrid = np.append(xGrid,v)
    idxU = idxU+adj if adj > 0 else idxU
    X = np.empty(N)
    x0 = xGrid[idxU-1]
    x1 = xGrid[idxU]
    x = x0[0]
    for i in range(N):
        x = root1(x,x1[i],fpnmc,U[i])
        X[i] = x
    return X

def root1(x0,x1,mu,u):    
    d = x1-x0
    sol = root_scalar(getFx,args=(mu,u),method='secant',x0=x0+0.35*d,x1=x0+0.67*d,fprime=getfx,
                       bracket=(x0,x1),options={'disp':False})  
    if sol.converged:
        return sol.root
    else:
        errorString = f'Did not converge on x0={x0},x1={x1},u={u}'
        raise Exception(errorString)
#%%
U1 = norm.cdf(rvu.ppf(U))
%timeit pnmc1 = xFromU(U1,Fx,xGrid,fpnmc)

#%%
u = U[3]
%timeit getFx(2,fpnmc,u)

def getFx(x,fpnmc,u):
    return x-u
%timeit sol = root_scalar(CMix,args=(U[i],),method='secant',x0=U[i],x1=U[i]*1.0001,bracket=(0,1),options={'disp':False})
%timeit sol = root_scalar(getFx,args=(fpnmc,u),method='brentq',x0=fpnmc[3],x1=fpnmc[4],fprime=getfx,bracket=(-100,100),options={'disp':False})  
%timeit brentq(CMix,0,1,args=(U[i],),disp=False)
pnmc2 = xFromU(U2,Fx,xGrid,fpnmc)
graphDensity3(t+1,nff,x1density,[pnmc1,pnmc2])
#%%
for t in range(1,T-1):
    t0 = time.time()
    xi = q(fmu(xi),N)
    pzx = rvz.pdf(Z[t]-xi)
    wi = wi*pzx#*pxx/qxz
    wi /= wi.sum()
    nff = 1/(wi@wi)
    if nff < N/4:
        xi = random.choice(xi,size=N,p=wi) # resample
        wi = Ninv     
    muXt = fmu(xi@wi)
    xGrid = muXt + Grid
    x1density = wi@rvx.pdf(tile(xGrid,(N,1))-tile(fmu(xi),(ndx,1)).T)
    
    U = random.uniform(size=N); U.sort()    
    fpnmc1 = fmu(pnmc1)
    Fx = rvx.cdf(tile(xGrid,(N,1))-tile(fpnmc1,(ndx,1)).T).mean(0)
    Fz = rvz.cdf(Z[t]-pnmc1).mean() 
    rvu = norm(loc=rho*norm.ppf(Fz),scale=rt_rho)
    U1 = norm.cdf(rvu.ppf(U))
    pnmc1 = xFromU(U1,Fx,xGrid,fpnmc1)
#    
#    alpha = getAlpha(t)
#    fpnmc2 = fmu(pnmc2)
#    Fx = rvx.cdf(tile(xGrid,(N,1))-tile(fpnmc2,(ndx,1)).T).mean(0)
#    Fz = rvz.cdf(Z[t]-pnmc2).mean() 
#    rvu = norm(loc=rho2*norm.ppf(Fz),scale=rt_rho2)
#    U2 = mixCop(U)
#    pnmc2 = xFromU(U2,Fx,xGrid,fpnmc2)
    t1 = time.time()
    print(f'{t}: {round(t1-t0,4)} seconds')
#    graphDensity3(t+1,nff,x1density,[pnmc1,pnmc2])
