module MatrixImpute

using Distributed: pmap, @distributed, @everywhere
using LinearAlgebra: pinv, svd, norm, I, dot, diagm
using StatsBase: sample
using Arpack: svds
using DataFrames: allowmissing
using Statistics: mean



function Impute(A,k;method=:fastImpute,γ = 10e4, λ=0.1, B=nothing, lr= 1/64)

    dims = length(size(A))

    if dims == 2
        if method == :fastImpute
            Aopt = fastImpute(A,k,γ,lr,B, approx = true)
            return Aopt
        else
            error("Method not yet implemented!")
        end
    elseif dims ==3
        if method == :fastImputeT && B == nothing
            Aopt = fastImputeT(A,k,γ,lr)
            return Aopt
        elseif method == :fastImputeT
            error("Tensor with side information is not supported!")
        else
            error("Method not yet implemented!")
        end
    else
        error("Number of Dimensions Incorrect")
    end
end


function fastImpute(A,k,γ,lr,B; approx = true)
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
        if nchunks>1
            for sc in startchunk
                UoptT, SoptT = fastImputeInner(A[:,sc:min(sc+chunk-1,p)],k,γ,lr,B)
                Aopt[:,sc:min(sc+chunk-1,p)] = UoptT * SoptT
            end
            if approx
                return Aopt
            else
                output = svds(Aopt, nsv = k)
                Aopt = output[1].U*diagm(output[1].S)*output[1].Vt
                return Aopt
            end
        else
            Uopt, Sopt = fastImputeInner(A,k,γ,lr,B)
            return Uopt * Sopt
        end
    else
        if m != size(B)[1]
            error("Sizes of A and B must match")
        end
        Uopt, Sopt = fastImputeInner(A,k,γ,lr,B)
        return Uopt * Sopt
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
        output = svds(A,nsv = k)
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
        return Arow - Xrow * (inv(I / γ + Xrow' * Xrow) * (Xrow' * Arow))
    end
    Wpar = Array{Float64}[W[samplen[i],samplem] for i=1:nnew]
    Xpar = Array{Float64}[S[:,samplem[Wpar[i].==1]]' for i=1:nnew]
    Apar = Array{Float64}[A[samplen[i],samplem[Wpar[i].==1]] for i=1:nnew]
    objpar = map(SmallInv,Xpar,Wpar,Apar)
    obj = @distributed (+) for i = 1:nnew
        dot(Apar[i],objpar[i]) / normfactor
    end
    Btemp2 = zeros(k,nnew)
    for i = 1:nnew
        Btemp2[:,i] = Xpar[i]' * objpar[i]
        ∇obj[:,samplem[Wpar[i].==1]] += -2 * γ * Btemp2[:,i] * objpar[i]' / normfactor
    end
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
        return Arow - Xrow * (inv(I / γ + Xrow' * Xrow) * (Xrow' * Arow))
    end
    Wpar = Array{Float64}[W[samplen[i],samplem] for i=1:nnew]
    Xpar = Array{Float64}[X[Wpar[i].==1,:] for i=1:nnew]
    Apar = Array{Float64}[A[samplen[i],samplem[Wpar[i].==1]] for i=1:nnew]
    objpar = map(SmallInv,Xpar,Wpar,Apar)
    obj = @distributed (+) for i = 1:nnew
        dot(Apar[i],objpar[i]) / (2*mnew*nnew)
    end
    Btemp1 = zeros(p,nnew)
    Btemp2 = zeros(k,nnew)
    for i = 1:nnew
        Btemp1[:,i] = B[samplem[Wpar[i].==1],:]' * objpar[i]
        Btemp2[:,i] = X[Wpar[i].==1,:]' * objpar[i]
    end
    ∇obj = @distributed (+) for i = 1:nnew
        -2 * γ * Btemp2[:,i] * Btemp1[:,i]' / (2*mnew*nnew)
    end
    return obj, ∇obj
end

function fastImputeT(A,k,γ,lr)
# Unknown values should be in missing for A
# A is the tensor that needs to be imputed, size n x m x p
# k is the maximum CP rank of the resultant matrix
# γ is the regularization parameter
# lr is the learning rate of the gradient descent function
    n,m,p = size(A)
    Amat = reshape(A,(n*m,p))
    W = ones(n,m,p)-ismissing.(A)
    Wmat = ones(n*m,p)-ismissing.(Amat)
    x = rand(n, k)
    y = rand(m, k)
    M = sum(ismissing.(A))/(n*m*p)

    z = [x; y]
    z = z ./ norm(z)
    x = z[1:n,:]
    y = z[n+1:end,:]
    i = 0
    j = 0
    println("Preprocessing Complete")
    obj0, ∇obj0x, ∇obj0y = TensorDeriv(Amat,Wmat,x,y,k,γ,M,j,size(A))

    ∇obj0 = [∇obj0x; ∇obj0y]
    ∇obj = zeros(m+n,k)

    objbest = Inf
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
        ∇tan = - ∇obj + dot(∇obj,z) * z
        updategrad = ∇tan / norm(∇tan)
        z = z .* cos(pi * lr) + updategrad * sin(pi * lr)
        z = z ./ norm(z)
        x = z[1:n,:]
        y = z[n+1:end,:]
        obj0, ∇obj0x, ∇obj0y = TensorDeriv(Amat,Wmat,x,y,k,γ,M,j,size(A))
        ∇obj0 = [∇obj0x; ∇obj0y]
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
        i = i + 1
    end
    X=zeros(m*n,k)
    function findbeta(Xrow, Arow)
        return inv(I/γ+Xrow'*Xrow)*Xrow'*Arow
    end
    for i=1:k
        X[:,i]=reshape(x[:,i]*y[:,i]',n*m)
    end
    function PopulateA(Xtemp,Atemp)
        return X*(pinv(Xtemp'*Xtemp,1e-7)*(Xtemp'*Atemp))
    end
    Aopt=zeros(n,m,p)
    Wpar=Array{Float64}[Wmat[:,i] for i=1:p]
    Xpar=Array{Float64}[X[Wpar[i].==1,:] for i=1:p]
    Apar=Array{Float64}[Amat[Wpar[i].==1,i] for i=1:p]
    result=map(PopulateA,Xpar,Apar)
    println("Model Solved")
    for i=1:p
        Aopt[:,:,i]=reshape(result[i],(n,m))
    end
    return Aopt
end

function TensorDeriv(Amat,Wmat,x,y,k,γ,M,j,dim)
    n,m,p = dim
    ncube = (n*m*p)^(1/3)
    nsamples = ncube * k * log(ncube)^2 * 4 * 2^j / (1-M)
    α = nsamples / (n*m*p)
    pnew = min(max(100, Int(round(α * p))), p)
    samplep = sample(1:p,pnew,replace=false)
    normfactor = n * m * pnew
    function SmallInv(Xrow, Wrow, Arow)
        return Arow- Xrow*(inv(I/γ+Xrow'*Xrow)*(Xrow'*Arow))
    end
    X = zeros(n*m,k)
    for i=1:k
        X[:,i] = reshape(x[:,i]*y[:,i]',n*m)
    end
    Wpar=Array{Float64}[Wmat[:,samplep[i]] for i=1:pnew]
    Xpar=Array{Float64}[X[Wpar[i].==1,:] for i=1:pnew]
    Apar=Array{Float64}[Amat[Wpar[i].==1,samplep[i]] for i=1:pnew]
    objpar=map(SmallInv,Xpar,Wpar,Apar)
    obj=@distributed (+) for i=1:pnew
        dot(Apar[i],objpar[i]) / normfactor
    end
    ∇objx=zeros(n,k)
    ∇objy=zeros(m,k)
    ∇obj=zeros(n*m,k)
    # Btemp1 = zeros(p, k)
    for i = 1:pnew
        # Btemp1[i, :] = objpar[i]' * Xpar[i]
        ∇obj[Wpar[i].==1,:] += -2 * γ * objpar[i] * (objpar[i]' * Xpar[i]) / normfactor
    end

    ∇obj = reshape(∇obj, (n,m,k))
    for i = 1:k
        ∇objx[:,i]=∇obj[:,:,i] * y[:,i]
    end
    for i = 1:k
        ∇objy[:,i]=∇obj[:,:,i]' * x[:,i]
    end

    return obj, ∇objx, ∇objy
end

export Impute

end # module
