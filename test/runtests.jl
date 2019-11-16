n=10^4
m=10^4
k=5
U=rand(Float64,n,k)
V=rand(Float64,k,m)
Afull=U*V
A=copy(Afull)
A = allowmissing(A)
A[rand(Float64,n,m).>0.05].=missing
t1 = time_ns()
Uopt, Sopt = MatrixImpute.Impute(A,k,γ = 10000)
println(mean(abs.(Uopt*Sopt-Afull)./abs.(Afull)))
println("Completed Test for Matrix Completion without Side Information")


n=10^3
m=10^3
p=100
k=5
U=rand(Float64,n,k)
R=rand(Float64,k,p)
B=rand(Float64,p,m)
Afull=U*R*B
A=copy(Afull)
A = allowmissing(A)
A[rand(Float64,n,m).>0.05].=missing
t1 = time_ns()
Uopt, Sopt = MatrixImpute.Impute(A,k,γ = 10000, B = B')
println(mean(abs.(Uopt*Sopt-Afull)./abs.(Afull)))
println("Completed Test for Matrix Completion with Side Information")
