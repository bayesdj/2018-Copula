from numpy import exp, log, random, array, arange, zeros, sqrt, ones
from numpy import tile, argmin, linspace, stack, outer
import numpy as np
import scipy.stats as stats
from scipy.stats import norm
from scipy.optimize import newton, root_scalar
import matplotlib.pyplot as plt
import timeit, time
from itertools import repeat
# one step ahead prediction, x_{k+1} | z_k
#%%
def fmu(x):
    return x-1

def q(mu,N):
    return mu + rvx.rvs(size=N)

def c(u,v):
    s = norm.ppf(u)
    t = norm.ppf(v)
    return rho2inv*exp(-0.5*(rho2*s*s-2*rho*s*t+rho2*t*t)*rho2inv*rho2inv)

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
    
def resample(xi,wi,N):
    return random.choice(xi,size=N,p=wi)

#%%
def getFx(x,mu,u):
    cdf = rvx.cdf(x-mu).mean()
    return cdf-u

def getfx(x,mu,u):
    return rvx.pdf(x-mu).mean()

def root1(x0,x1,mu,u):    
    sol = root_scalar(getFx,args=(mu,u),method='brenth',
                       bracket=(x0,x1),options={'disp':False})
#    sol = root_scalar(getFx,args=(mu,u),method='newton',x0=0.5*(x0+x1),
#                       fprime=getfx,options={'disp':False})    
    return sol.root if sol.converged is True else np.NaN
#    return brenth(getFx,x0,x1,args=(mu,u))
#%%
def buildRev(xGrid,fmupnmc,Fx,Fz):
    N = len(fmupnmc);
    U = random.uniform(size=N)
    U = norm.ppf(U,loc=rho*norm.ppf(Fz),scale=sqrt(1-rho2))
    U = norm.cdf(U)

    idxU = np.searchsorted(Fx,U)
    #xHi = xGrid[-1] + 0.1*abs(xGrid[-1])
    #fxmcSorted = np.append(panel[:,0],xHi)
    adj = 0
    m = 10
    if idxU.min() == 0:
        v = xGrid[0]-abs(xGrid[0])*m
        xGrid = np.insert(xGrid,0,v)
        adj += 1
    if idxU.max() == ndx:
        v = xGrid[-1]+abs(xGrid[-1])*m
        xGrid = np.append(xGrid,v)
    idxU = idxU+adj if adj > 0 else idxU
    x0 = xGrid[idxU-1]
    x1 = xGrid[idxU]
    #g = root1(x0[0],x1[0],U[0],fmupnmc)
    t0 = time.time()
    g = map(root1,x0,x1,repeat(fmupnmc,N),U)
    X = np.fromiter(g,dtype=np.float,count=N)    
    t1 = time.time()
    print(f'{round(t1-t0,4)} seconds')
    return X

def buildRev2(xGrid,fmupnmc,Fx,Fz):
    # 24-28 seconds per 10k
    N = len(fmupnmc);
    U = random.uniform(size=N)
    U = norm.ppf(U,loc=rho*norm.ppf(Fz),scale=sqrt(1-rho2))
    U = norm.cdf(U)
    U.sort()

    idxU = np.searchsorted(Fx,U)
    #xHi = xGrid[-1] + 0.1*abs(xGrid[-1])
    #fxmcSorted = np.append(panel[:,0],xHi)
    adj = 0
    m = 10
    if idxU.min() == 0:
        v = xGrid[0]-abs(xGrid[0])*m
        xGrid = np.insert(xGrid,0,v)
        adj += 1
    if idxU.max() == ndx:
        v = xGrid[-1]+abs(xGrid[-1])*m
        xGrid = np.append(xGrid,v)
    idxU = idxU+adj if adj > 0 else idxU
    X = np.empty(N)
    x0 = xGrid[idxU-1]
    x1 = xGrid[idxU]
    x = x0[0]
    t0 = time.time()
    for i in range(N):
        x = root1(x,x1[i],fmupnmc,U[i])
        X[i] = x
    t1 = time.time()
    print(f'{round(t1-t0,4)} seconds')
    return X

def buildRev3(xGrid,fmupnmc,Fx,Fz):
    N = len(fmupnmc);
    U = random.uniform(size=N)
    U = norm.ppf(U,loc=rho*norm.ppf(Fz),scale=sqrt(1-rho2))
    U = norm.cdf(U)
    U.sort()
    
    x = rvx.ppf(U[0]).mean()+fmupnmc
    x = xGrid.min()    
# 21-22 seconds per 10k
    X = np.empty(N)   
    t0 = time.time()
    for i in range(N):
        x = newton(getFx,x,fprime=getfx,args=((fmupnmc,U[i])))
        X[i] = x
    t1 = time.time()
    print(f'{round(t1-t0,4)} seconds')
    return X

#%%
a = 1; b = 1
varx = a/b**2
sdx = sqrt(varx)
varz = 4
sdz = sqrt(varz)

rho = 0.5
rho2 = rho*rho
rho2inv = 1/sqrt(1-rho2)

rvx = stats.gamma(a,scale=b)
rvz = norm(scale=sdz)   

T = 50
N = int(1e4)
k = 7
ndx = 2**10+1
fxyMax = rvx.pdf(0) 
rvu = stats.uniform(0,fxyMax)
#%%
X = zeros(T)
Z = zeros(T)
Thi = zeros(T)
Mu = zeros(T)

x = rvx.rvs()
X[0] = x
z = x + rvz.rvs()
Z[0] = z
thi = varx*varz/(varx+varz)
mu = z/varz*thi
Thi[0] = thi
Mu[0] = mu

for t in range(1,T):    
    x = fmu(x)+rvx.rvs()
    z = x+rvz.rvs()
    varx2 = varx+thi
    thi = varz*varx2/(varz+varx2)
    mu = (z/varz + fmu(mu)/varx2)*thi
    X[t] = x
    Z[t] = z
    Thi[t] = thi
    Mu[t] = mu

Thi = sqrt(Thi+varx)
#%%
# particle filter 1
Ninv = 1/N*ones(N)
xi = q(0,N)
t = 0
wi = rvz.pdf(Z[t]-xi)
wi /= wi.sum()
nff = 1/(wi@wi)
muXt = fmu(xi@wi)
xGrid = linspace(muXt-k*sdx,muXt+k*sdx,ndx,retstep=False)
x1density = wi@rvx.pdf(tile(xGrid,(N,1))-tile(fmu(xi),(ndx,1)).T)
# kalman = norm.pdf(xGrid,fmu(Mu[t]),Thi[t])
#%%
pnmc = xi.copy()
#muxt = fmu(pnmc.mean())
#end0 = muxt-k*sdx; end1 = muxt+k*sdx
#xGrid = linspace(end0,end1,ndx,retstep=False)
fmupnmc = fmu(pnmc)
Fx = rvx.cdf(tile(xGrid,(N,1))-tile(fmupnmc,(ndx,1)).T).mean(0)
Fz = rvz.cdf(Z[t]-pnmc).mean()
pnmc = buildRev2(xGrid,fmupnmc,Fx,Fz)

graphDensity2(t+1,nff,x1density,pnmc)
#%%
for t in range(1,T-1):
    xi = q(fmu(xi),N)
    pzx = rvz.pdf(Z[t]-xi)
    wi = wi*pzx#*pxx/qxz
    wi /= wi.sum()
    nff = 1/(wi@wi)
    if nff < N/4:
        xi = resample(xi,wi,N)
        wi = Ninv     
    muXt = fmu(xi@wi)
    xGrid = linspace(muXt-k*sdx,muXt+k*sdx,ndx,retstep=False)
    x1density = wi@rvx.pdf(tile(xGrid,(N,1))-tile(fmu(xi),(ndx,1)).T)
    #kalman = norm.pdf(xGrid,fmu(Mu[t]),Thi[t])    
    fmupnmc = fmu(pnmc)
    Fx = rvx.cdf(tile(xGrid,(N,1))-tile(fmupnmc,(ndx,1)).T).mean(0)
    Fz = rvz.cdf(Z[t]-pnmc).mean()
    pnmc = buildRev2(xGrid,fmupnmc,Fx,Fz)
    graphDensity2(t+1,nff,x1density,pnmc)