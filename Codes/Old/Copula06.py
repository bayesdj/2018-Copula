from numpy import exp, log, random, array, arange, zeros, sqrt, ones
from numpy import tile, argmin, linspace, stack
from scipy.integrate import trapz, cumtrapz, romb
import scipy.stats as stats
from scipy.stats import norm, cauchy
import matplotlib.pyplot as plt
#%%
T = 19
varx = 1
sdx = sqrt(varx)
varz = 4
sdz = sqrt(varz)

X = zeros(T)
Z = zeros(T)
W = Z.copy()
Mu = Z.copy()

#random.seed(46)
x = random.standard_cauchy()
X[0] = x
z = random.normal(x,sdz)
Z[0] = z
w = 0; mu = 0
W[0] = w; Mu[0] = mu

rho = 4
rho2 = rho*rho
rho2inv = 1/sqrt(1-rho2)

def fmu(x):
    return x+1

def c(u,v):
    ea = exp(-rho)-1
    top = rho*-ea*exp(-rho*(u+v))
    bot = ea+(exp(-rho*u)-1)*(exp(-rho*v)-1)
    return top/(bot*bot)
#%%

for t in range(1,T):    
    x = fmu(x)+random.standard_t(2)
    z = random.normal(x,sdz)
    varx1 = varx+w
    w = varz*varx1/(varz+varx1)
    mu = w*(z/varz+(1+mu)/varx1)
    X[t] = x
    Z[t] = z
    W[t] = w
    Mu[t] = mu
    
#%%
k = 9 
nx = 2**11+1
Fmin = 1e-16
Fmax = 1-Fmin
    
sdx1 = sqrt(2)
xGrid,dx0 = linspace(1-k*sdx,1+k*sdx,nx,retstep=True)
fx = cauchy.pdf(xGrid)
Fx = cauchy.cdf(xGrid)
Fz = norm.cdf(Z[0],0,sqrt(5))
copula = c(Fx,Fz)
pn = copula*fx

#%%
for t in range(1,T-1):
    muxt = fmu(X[t])    
    x1Grid,dx1 = linspace(muxt-k*sdx,muxt+k*sdx,nx,retstep=True)
    x1Grid = tile(x1Grid,(nx,1))
    x0Grid = tile(fmu(xGrid),(nx,1))
    integrand = cauchy.pdf(x1Grid.T)*pn
    fx = romb(integrand,dx=dx0,axis=1) # numerical integration
    Fx = cumtrapz(fx,dx=dx1,initial=Fmin) # numerical integration
    Fx[Fx>=Fmax] = Fmax
    
    integrand = norm.cdf(Z[t],xGrid,sdz)*pn
    Fz = romb(integrand,dx=dx0) # numerical integration
    pn = c(Fx,Fz)*fx
    xGrid = x1Grid[0]
    dx0 = dx1
    pnTrue = norm.pdf(xGrid,fmu(Mu[t]),sqrt(W[t]+varx))
    
    plt.figure()
    title = f'$p{t+1}$'
    plt.plot(xGrid,stack([pn,pnTrue]).T)
    plt.axvline(x=X[t+1],color='black')
    plt.title(title)
    plt.legend([rf'copula $\rho$={rho}','Kalman'])
#%%
#plt.plot(z0Grid[0],fz)
#plt.axvline(x=Z[t],color='black')
#plt.title('f(z)')