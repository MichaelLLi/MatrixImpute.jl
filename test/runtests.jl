n=10^3
m=10^3
k=5
U=rand(Float64,n,k)
V=rand(Float64,k,m)
Afull=U*V
A=copy(Afull)
A = allowmissing(A)
A[rand(Float64,n,m).>0.05].=missing
t1 = time_ns()
Aopt = Impute(A,k,γ = 10000)
println("Completed Test for Matrix Completion without Side Information")
