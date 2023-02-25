import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

xx = np.linspace(1e-4,1,endpoint=False,num=5000)

u = 0.8
rho = 0.5
def cop(v):
    s = norm.ppf(u)
    t = norm.ppf(v)
    rho2 = rho*rho
    c = 1/np.sqrt(1-rho2)
    m = norm.cdf(s/rho)
    return m, c*np.exp(-c*c*0.5*(s*s*rho2+rho2*t*t-2*rho*s*t))

m, pdf = cop(xx)
plt.plot(xx,pdf)
plt.axvline(x=m,color='black')