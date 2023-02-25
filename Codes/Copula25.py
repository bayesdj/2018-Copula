from numpy import exp, log, random, array, arange, zeros, sqrt, ones
from numpy import tile, argmin, linspace, stack, outer
import numpy as np
import scipy.stats as stats
from scipy.stats import norm
from scipy.special import ndtri,ndtr
from scipy.optimize import root_scalar, brentq
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
#%%
def CMix(u1,u):
    x = ndtri(u1)
    z = (x-zloc)/zscale
    return alpha1*u1+alpha*ndtr(z)-u

#def cMix(x,u):
#    s = norm.ppf(x)
#    return alpha*(exp(rvu.logpdf(s)-norm.logpdf(s))-1)

def getFx(x,mu,u):
    z = (x-mu)/xscale
#    z = x-mu
    cdf = rvxx._cdf(z).mean()
    return cdf-u
#%%
#%timeit getFx(0,pnmc,0.3)
#%timeit CMix(0,0.2)
#%%
def getfx(x,mu,u):
    return rvxx._pdf(x-mu).mean()

def root1(x0,x1,mu,u):    
    #d = x1-x0
    sol = root_scalar(getFx,args=(mu,u),method='brentq',x0=x0,x1=x1,
                      bracket=(x0,x1),fprime=getfx,options={'disp':False})  
    if sol.converged:
        return sol.root
    else:
        errorString = f'Did not converge on x0={x0},x1={x1},u={u}'
        raise Exception(errorString)

def getAlpha(t):
    #return (1/(2+t))**0.65
    return 0.1

def mixCop(U):
    U2 = np.empty(N) 
    x = 0
    for i in range(0,N,2):
        x = brentq(CMix,x,1,args=(U[i],),disp=False)
#        x = root_scalar(CMix,args=(U[i],),x0=x,x1=1,method='brentq',bracket=(x,1),options={'disp':False}).root
        U2[i] = x
    if N%2 == 0:
        U2[-1] = brentq(CMix,x,1,args=(U[-1],),disp=False)
    for i in range(1,N-1,2):
        #U2[i] = brentq(CMix,U2[i-1],U2[i+1],args=(U[i],),disp=False)
        a = U2[i-1]; b = U2[i+1]-a
        U2[i] = root_scalar(CMix,args=(U[i],),x0=a+0.35*b,x1=a+0.67*b,
          method='secant',bracket=(a,a+b),options={'disp':False}).root
    return U2

g = np.vectorize(root1,otypes=[np.float64],doc='vt_root_1',excluded={2})
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
#    X = g(x0,x1,fpnmc,U)
    return X
#%%
a = 5; b = 1
#varx = (a*b)/((a+b)**2*(a+b+1))
#sdx = sqrt(varx)
#rvx = stats.beta(a,b,loc=-a/(a+b))  
aaa = 1; b = 1
#varx = aaa*b**2
#sdx = sqrt(varx)
rvx = stats.gamma(aaa,scale=b,loc=-a*b)
rvxx = stats.gamma
varx = 1
sdx = sqrt(varx)
#rvx = norm(scale=sdx)
#rvxx = stats.norm
#rvx = stats.t(df=5)
#rvxx = stats.t
xscale = sdx

a = 5; b = 1
varz = (a*b)/((a+b)**2*(a+b+1))
sdz = sqrt(varz)
rvz = stats.beta(a,b,loc=-a/(a+b))  
#rvz = norm(scale=sqrt(2))
#a = 1; b = 0.5
#varz = a*b**2
#sdz = sqrt(varz)
#rvz = stats.gamma(a,scale=b,loc=-a*b)
#rvz = stats.t(df=9) #norm(scale=sdz) 


rho = 0.48
rt_rho = sqrt(1-rho*rho)
rho2 = 0.7
rt_rho2 = sqrt(1-rho2*rho2)

T = 10
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
x1density = wi@rvxx._pdf(tile(xGrid,(N,1))-tile(fmu(xi),(ndx,1)).T)
#%%
pnmc = xi.copy()
fpnmc = fmu(pnmc)
Fx = rvxx._cdf(tile(xGrid,(N,1))-tile(fpnmc,(ndx,1)).T).mean(0)
Fz = rvz.cdf(Z[t]-pnmc).mean()
zloc,zscale = rho*ndtri(Fz),rt_rho
#rvu = norm(loc=rho*ndtri(Fz),scale=rt_rho)
U = random.uniform(size=N); U.sort()
#U1 = norm._cdf(rvu.ppf(U))
U1 = ndtr(ndtri(U)*zscale+zloc)
pnmc1 = xFromU(U1,Fx,xGrid,fpnmc)

zloc,zscale = rho2*ndtri(Fz),rt_rho2
U2 = mixCop(U)
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
    x1density = wi@rvxx._pdf(tile(xGrid,(N,1))-tile(fmu(xi),(ndx,1)).T)
    U = random.uniform(size=N); U.sort()
    
    fpnmc1 = fmu(pnmc1)
    Fx = rvxx._cdf(tile(xGrid,(N,1))-tile(fpnmc1,(ndx,1)).T).mean(0)
    Fz = rvz.cdf(Z[t]-pnmc1).mean() 
    #rvu = norm(loc=rho*ndtri(Fz),scale=rt_rho)
    zloc,zscale = rho*ndtri(Fz),rt_rho
    U1 = ndtr(ndtri(U)*zscale+zloc)
    pnmc1 = xFromU(U1,Fx,xGrid,fpnmc1)
    
    alpha = getAlpha(t)
    fpnmc2 = fmu(pnmc2)
    Fx = rvxx._cdf(tile(xGrid,(N,1))-tile(fpnmc2,(ndx,1)).T).mean(0)
    Fz = rvz.cdf(Z[t]-pnmc2).mean() 
    #rvu = norm(loc=rho2*ndtri(Fz),scale=rt_rho2)
    zloc,zscale = rho2*ndtri(Fz),rt_rho2
    U2 = mixCop(U)
    pnmc2 = xFromU(U2,Fx,xGrid,fpnmc2)
    t1 = time.time()
    print(f'{t}: {round(t1-t0,4)} seconds')
    graphDensity3(t+1,nff,x1density,[pnmc1,pnmc2])
