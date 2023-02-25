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
#    return mu+random.normal(scale=sd,size=N)
    return mu + rv.rvs(size=N)

def graphDensity(t,xi,wi,nff):
    nff = round(nff,1)
    title = f'$p{t}$, NFF={nff}'
    plt.figure()
    plt.bar(xi,wi,width=0.5)
    plt.axvline(x=X[t],color='black')
    plt.title(title)
#%%
df = 6   
rv = stats.t(df=df)
    
T = 20
varx = 1
sdx = sqrt(varx)
varz = 4
sdz = sqrt(varz)

#%%
X = zeros(T)
Z = zeros(T)

x = random.standard_t(df)
X[0] = x
z = random.normal(x,sdz)
Z[0] = z

for t in range(1,T):    
    x = fmu(x)+random.standard_t(df)
    z = x+random.normal(0,sdz)
    X[t] = x
    Z[t] = z
    
#%%
N = int(5e3)
xt = q(0,1)
wi = norm.pdf(Z[0],loc=xt,scale=sdz)
wi /= wi.sum()
nff = 1/(wi@wi)
graphDensity(0,xt,wi,nff)
#%%
for t in range(1,T):
    xtmu = fmu(xt)
    xt = q(xtmu,1)
    pzx = norm.pdf(Z[t],loc=xt,scale=sdz)
#    pxx = stats.t.pdf(xi,loc=fmu(x0),df=df)
#    qxz = norm.pdf(xi,loc=qmean,scale=qsd)
    wi = wi*pzx#*pxx/qxz
    wi /= wi.sum()
    nff = 1/(wi@wi)
    
    graphDensity(t,xt,wi,nff)