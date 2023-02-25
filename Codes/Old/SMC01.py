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

def q(mu,sd):
    #return mu+random.standard_t(df=df,size=N) #normal(scale=sd,size=N)
    return mu+random.normal(scale=sd,size=N)

def graphDensity(t,xi,wi):
    title = f'$p{t}$'
    plt.figure()
    plt.bar(xi,wi,width=0.5)
    plt.axvline(x=X[t],color='black')
    plt.title(title)
#%%
rho = 0.5
rho2 = rho*rho
rho2inv = 1/sqrt(1-rho2)
df = 6    
    
T = 20
varx = 1
sdx = sqrt(varx)
varz = 4
sdz = sqrt(varz)

#%%
X = zeros(T)
Z = zeros(T)
W = Z.copy()
Mu = Z.copy()

x = random.standard_t(df)
X[0] = x
z = random.normal(x,sdz)
Z[0] = z
w = 0; mu = 0
W[0] = w; Mu[0] = mu

for t in range(1,T):    
    x = fmu(x)+random.standard_t(df)
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
xgrid = linspace(-k,k,nx)
#%%
N = int(1e3)
x0 = X[0]*ones(N)

qmean = qmu(x0,Z[0])
qsd = sdz + 1
xi = q(qmean,qsd)


pzx = norm.pdf(Z[0],loc=x0,scale=sdz)
pxx = stats.t.pdf(xi,loc=fmu(x0),df=df)
#qxz = stats.t.pdf(xi,loc=qmean,df=df)
qxz = norm.pdf(xi,loc=qmean,scale=qsd)
wi = pzx*pxx/qxz
wi /= wi.sum()

#%%
h = plt.bar(xi,wi,width=0.5)
plt.axvline(x=X[1],color='black')
#%%
for t in range(1,T-1):
    x0 = xi
    qmean = qmu(x0,Z[t])
    xi = q(qmean,qsd)
    pzx = norm.pdf(Z[0],loc=x0,scale=sdz)
    pxx = stats.t.pdf(xi,loc=fmu(x0),df=df)
    qxz = norm.pdf(xi,loc=qmean,scale=qsd)
    wi = wi*pzx*pxx/qxz
    wi /= wi.sum()
    graphDensity(t+1,xi,wi)