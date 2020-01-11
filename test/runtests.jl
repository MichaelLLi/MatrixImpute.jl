using DataFrames, TensorToolbox, MatrixImpute

n=10^5
m=10^3
k=7
U=rand(Float64,n,k)
V=rand(Float64,k,m)
Afull=U*V
A=copy(Afull)
A = allowmissing(A)
A[rand(Float64,n,m).>0.05].=missing
t1 = time_ns()
Aopt = MatrixImpute.Impute(A,k,γ = 10000)
println("Completed Test for Matrix Completion without Side Information")


n=10^3
m=10^3
p=100
k=3
U=rand(Float64,n,k)
R=rand(Float64,k,p)
B=rand(Float64,p,m)
Afull=U*R*B
A=copy(Afull)
A = allowmissing(A)
A[rand(Float64,n,m).>0.05].=missing
t1 = time_ns()
Aopt = MatrixImpute.Impute(A,k,γ = 10000, B = B')
println("Completed Test for Matrix Completion with Side Information")

n=100
m=200
p=500
k=6
M=0.8
γ=1000000
stepsize=64
A=rand(n,m,p)
A=allowmissing(A)
A[rand(n,m,p).>1-M].=missing
t1 = time_ns()
Aopt = MatrixImpute.Impute(A,k,γ = γ, method = :fastImputeT)
println("Completed Test for Tensor Completion")
