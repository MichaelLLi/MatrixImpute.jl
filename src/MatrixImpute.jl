module MatrixImpute

using Distributed: pmap, @distributed, @everywhere
using LinearAlgebra: pinv, svd, norm, I, dot
using StatsBase: sample
using Arpack: svds
using DataFrames: allowmissing



function Impute(A,k;method=:fastImpute,γ = 10e4, λ=0.1, B=nothing, lr= 1/64)

if method == :fastImpute
    Aopt = fastImpute(A,k,γ,lr,B)
    return Aopt
else
    error("Method not yet implemented!")
end

end


function fastImpute(A,k,γ,lr,B)
# Unknown values should be in missing for A
# A is the matrix that needs to be imputed, size n x m
# B is the feature selector, size m x p
# k is the maximum rank of the resultant matrix
# γ is the regularization parameter
# lr is the learning rate of the gradient descent function
    n = size(A)[1]
    m = size(A)[2]
    chunk = 1000
    if B == nothing
        p = m
        startchunk = collect(1:chunk:m)
        nchunks = length(startchunk)
        Uopt = zeros(Float64,n,k)
        Sopt = zeros(Float64,k,p)
        Aopt = zeros(Float64,n,p)
        t=0
        for sc in startchunk
            UoptT, SoptT = fastImputeInner(A[:,sc:min(sc+chunk-1,p)],k,γ,lr,B)
            Aopt[:,sc:min(sc+chunk-1,p)] = UoptT * SoptT
            Sopt[:,sc:min(sc+chunk-1,p)] = SoptT
        end
        return Aopt
    else
        if m != size(B)[1]
            error("Sizes of A and B must match")
        end
        Uopt, Sopt = fastImputeInner(A,k,γ,lr,B)
        return Uopt, Sopt
    end
end


function fastImputeInner(A,k,γ,lr,B)
    n = size(A)[1]
    m = size(A)[2]
    if B == nothing
        p = m
    elseif B != nothing
        p = size(B)[2]
    end
    M = sum(ismissing.(A))/(n*m)
    Uopt = zeros(n,k)
    W = zeros(n,m)
    for i = 1:n
        W[i,:] = ones(m) - convert(Vector{Float64},ismissing.(A[i,:]))
    end
    println("Preprocessing Complete")
    A[ismissing.(A)] .= 0
    if M<0.5 && B == nothing
        output = svds(A,nsv=5)
        S = diagm(output[1].S)*output[1].Vt
    else
        S = rand(k,p)
        S = S ./ norm(S)
    end
    objbest = Inf
    ∇obj = zeros(k,p)
    i = 0
    j = 0
    if B != nothing
        obj0, ∇obj0 = MatrixDeriv2(A,B,W,S,k,γ,M,j)
    else
        obj0, ∇obj0 = MatrixDeriv(A,W,S,k,γ,M,j)
    end
    noimprov = 0
    raised = false
    while i<50
        if (noimprov>=5) && (!raised)
            j = j + 1
            raised = true
            objbest = obj0
            noimprov = 0
        end
        ∇obj = ∇obj0 + i / (i+3) * ∇obj
        ∇tan = -∇obj + dot(∇obj,S) * S
        updategrad = ∇tan / norm(∇tan)
        S = S * cos(pi * lr) + updategrad*sin(pi * lr)
        S = S / norm(S)
        if B != nothing
            obj0, ∇obj0 = MatrixDeriv2(A,B,W,S,k,γ,M,j)
        else
            obj0, ∇obj0 = MatrixDeriv(A,W,S,k,γ,M,j)
        end
        if obj0 > objbest
            noimprov = noimprov + 1
            if (noimprov % 5 == 0) && (raised)
                raised = false
            end
        else
            noimprov = 0
            objbest = obj0
            raised = false
        end
        println("There are $noimprov no improvement iterations")
        println("Estimated objective is: $obj0")
        i = i + 1
    end
    Sopt = S
    if B == nothing
        X = Sopt'
    else
        X = B * Sopt'
    end
    println("Model Solved")
    function PopulateA(Xtemp,Atemp)
        return pinv(Xtemp'*Xtemp,1e-7)*(Xtemp'*Atemp)
    end
    Xtemp=Array{Float64}[X[W[i,:].==1,:] for i=1:n]
    Atemp=Array{Float64}[A[i,W[i,:].==1] for i=1:n]
    result=pmap(PopulateA,Xtemp,Atemp)
    for i=1:n
        Uopt[i,:]=result[i]
    end
    return Uopt, Sopt
end

function MatrixDeriv(A,W,S,k,γ,M,j)
    # Derivative Calculation for Matrix Completion without Side Information
    n = size(A)[1]
    m = convert(Int,size(A)[2])
    p = m
    nsquare = sqrt(n*m)
    mnew = m
    nnew = min(max(100,Int(round(nsquare * log(nsquare) * k * 2^j/(8 * mnew * (1-M))))),n)
    normfactor = 2 * mnew * nnew
    ∇obj = zeros(k,p)
    samplen=sample(1:n,nnew,replace=false)
    samplem=sample(1:m,mnew,replace=false)
    obj=0
    SmallInv = function (Xrow, Wrow, Arow)
        return (I- Xrow * inv(I / γ + Xrow' * Xrow) * Xrow') * Arow
    end
    Wpar = Array{Float64}[W[samplen[i],samplem] for i=1:nnew]
    Xpar = Array{Float64}[S[:,samplem[Wpar[i].==1]]' for i=1:nnew]
    Apar = Array{Float64}[A[samplen[i],samplem[Wpar[i].==1]] for i=1:nnew]
    objpar = map(SmallInv,Xpar,Wpar,Apar)
    obj = @distributed (+) for i = 1:nnew
        dot(Apar[i],objpar[i]) / normfactor
    end
    println("Starting DerivCalc")
    println("We chose $nnew n")
    println("We chose $mnew m")
    Btemp2 = zeros(k,nnew)
    for i = 1:nnew
        Btemp2[:,i] = Xpar[i]' * objpar[i]
        ∇obj[:,samplem[Wpar[i].==1]] += -2 * γ * Btemp2[:,i] * objpar[i]' / normfactor
    end
    println("Derivative Calculated")
    return obj, ∇obj
end

function MatrixDeriv2(A,B,W,S,k,γ,M,j)
    # Derivative Calculation for Matrix Completion with Side Information
    p = size(B)[2]
    n = size(A)[1]
    m = convert(Int,size(A)[2])
    ∇obj = zeros(k,p)
    nsquare = sqrt(n * m)
    mnew = min(4*p,m)
    nnew = min(max(100,Int(round(nsquare * log(nsquare) * k * 2^j / (8 * mnew * (1-M))))),n)
    samplen = sample(1:n,nnew,replace=false)
    samplem = sample(1:m,mnew,replace=false)
    X = B[samplem,:]*S'
    obj = 0
    SmallInv = function (Xrow, Wrow, Arow)
        return (I- Xrow*inv(I/γ+Xrow'*Xrow)*Xrow')*Arow
    end
    Wpar = Array{Float64}[W[samplen[i],samplem] for i=1:nnew]
    Xpar = Array{Float64}[X[Wpar[i].==1,:] for i=1:nnew]
    Apar = Array{Float64}[A[samplen[i],samplem[Wpar[i].==1]] for i=1:nnew]
    objpar = map(SmallInv,Xpar,Wpar,Apar)
    obj = @distributed (+) for i = 1:nnew
        dot(Apar[i],objpar[i]) / (2*mnew*nnew)
    end
    println("Starting DerivCalc")
    println("We chose $nnew n")
    println("We chose $mnew m")
    Btemp1 = zeros(p,nnew)
    Btemp2 = zeros(k,nnew)
    for i = 1:nnew
        Btemp1[:,i] = B[samplem[Wpar[i].==1],:]' * objpar[i]
        Btemp2[:,i] = X[Wpar[i].==1,:]' * objpar[i]
    end
    ∇obj = @distributed (+) for i = 1:nnew
        -2 * γ * Btemp2[:,i] * Btemp1[:,i]' / (2*mnew*nnew)
    end
    println("Derivative Calculated")
    return obj, ∇obj
end

export Impute

end # module
