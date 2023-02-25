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

def c(u,v):
    s = stats.t.ppf(u,df)
    t = stats.t.ppf(v,df)
    left = gamma(0.5*(df+2))*gamma(0.5*df)/gamma(0.5*(df+1))**2*rho2inv
    right = ((1+s*s/df)*(1+t*t/df))**(0.5*(df+1))/(1+(s*s+t*t-2*rho*s*t)/df*rho2inv*rho2inv)**(0.5*(df+2))
    return left*right

def graphDensity(t,xGrid,fx):
    title = f'$p{t}$'
    plt.figure()
    plt.plot(xGrid,stack([pn,pnTrue]).T)
    plt.axvline(x=X[t],color='black')
    plt.title(title)
    plt.legend([rf'copula $\rho$={rho}','Kalman'])
#%%
rho = 0.5
rho2 = rho*rho
rho2inv = 1/sqrt(1-rho2)
df = 3    
    
T = 50
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
    
#sdx1 = sqrt(2)
xGrid,dx0 = linspace(X[1]-k*sdx,X[1]+k*sdx,nx,retstep=True)
fx = stats.t.pdf(xGrid,df=df)
Fx = stats.t.cdf(xGrid,df=df)
Fz = romb(norm.cdf(Z[0],xGrid,sdz)*fx,dx=dx0)
pn = c(Fx,Fz)*fx
pnTrue = norm.pdf(xGrid,fmu(Mu[0]),sqrt(W[0]+varx))
graphDensity(1,xGrid,stack([pn,pnTrue]).T)

#%%
for t in range(1,T-1):
    muxt = fmu(X[t])    
    x1Grid,dx1 = linspace(muxt-k*sdx,muxt+k*sdx,nx,retstep=True)
    x1Grid = tile(x1Grid,(nx,1))
    x0Grid = tile(fmu(xGrid),(nx,1))
    integrand = stats.t.pdf(x1Grid.T-x0Grid,df=df)*pn
    fx = romb(integrand,dx=dx0,axis=1) # numerical integration
    Fx = cumtrapz(fx,dx=dx1,initial=Fmin) # numerical integration
    Fx[Fx>=Fmax] = Fmax
    
    integrand = norm.cdf(Z[t],xGrid,sdz)*pn
    Fz = romb(integrand,dx=dx0) # numerical integration
    pn = c(Fx,Fz)*fx
    xGrid = x1Grid[0]
    dx0 = dx1
    pnTrue = norm.pdf(xGrid,fmu(Mu[t]),sqrt(W[t]+varx))
    
    graphDensity(t+1,xGrid,stack([pn,pnTrue]).T)
#%%
#plt.plot(z0Grid[0],fz)
#plt.axvline(x=Z[t],color='black')
#plt.title('f(z)')