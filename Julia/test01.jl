# using LinearAlgebra
# A = rand(3, 3)
# x = fill(1, (3,))
# b = A * x
#
# Alu = lu(A)
# using Pkg
# Pkg.add("Plots")
using Plots
##
p = Array(0:1e-3:1)

function f(p,n)
    return p.^(n-1).*(1 .- p)
end

n = 11
pp = f(p,n)
##
plot(p,pp)
#---
mxval,mxix = findmax(pp)
opt = p[mxix]
opt1 = (n-1)/n
#---
