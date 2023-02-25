from numpy import exp, log, random, array, arange, zeros, sqrt, ones
from numpy import tile, argmin, linspace, stack
from scipy.integrate import trapz, cumtrapz, romb
from scipy.special import gamma
import scipy.stats as stats
from scipy.stats import norm, cauchy
import matplotlib.pyplot as plt
#%%
def fmu(x):
    return x+1

def qmu(x,z):
    return 0.5*(fmu(x)+fmu(z))

def q(mu,N):
    return mu + rvx.rvs(size=N)

def graphDensity(t,xi,wi,nff,density):
    nff = round(nff,1)
    title = f'$p{t}$, '+r'$N_{eff}$ = '+f'{nff}'
    plt.figure()
    plt.plot(xGrid,density)
    plt.hist(xi,weights=wi,density=True,bins=250)
    plt.axvline(x=X[t],color='black')
    plt.title(title)
    
def resample(xi,wi,N):
    return random.choice(xi,size=N,p=wi)
#%%
T = 100
varx = 1
sdx = sqrt(varx)
varz = 4
sdz = sqrt(varz)

rvx = stats.norm(scale=sdx)
rvz = stats.norm(scale=sdz)    

k = 6
nx = 2**10+1
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
    
#%%
N = int(1e4)
Ninv = 1/N
xi = q(0,N)
wi = norm.pdf(Z[0],loc=xi,scale=sdz)
wi /= wi.sum()
nff = 1/(wi@wi)
muXt = xi@wi
xGrid = linspace(muXt-k*sdx,muXt+k*sdx,nx,retstep=False)
kalman = norm.pdf(xGrid,Mu[0],sqrt(Thi[0]))
graphDensity(0,xi,wi,nff,kalman)
#%%
for t in range(1,T):
    xtmu = fmu(xi)
    xi = q(xtmu,N)
    pzx = norm.pdf(Z[t],loc=xi,scale=sdz)
    wi = wi*pzx#*pxx/qxz
    wi /= wi.sum()
    nff = 1/(wi@wi)
    if nff < N/4:
        xi = resample(xi,wi,N)
        wi = Ninv*ones(N)
        
    muXt = xi@wi
    xGrid = linspace(muXt-k*sdx,muXt+k*sdx,nx,retstep=False)
    kalman = norm.pdf(xGrid,Mu[t],sqrt(Thi[t]))
    graphDensity(t,xi,wi,nff,kalman)