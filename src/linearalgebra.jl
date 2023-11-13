# general Toeplitz

# Fast application of a general Toeplitz matrix to a column vector via FFT
function mul!(
    y::StridedVector, A::AbstractToeplitz, x::StridedVector, α::Number, β::Number
)
    m, n = size(A)
    if length(y) != m
        throw(DimensionMismatch(
            "first dimension of A, $(m), does not match length of y, $(length(y))"
        ))
    end
    if length(x) != n
        throw(DimensionMismatch(
            "second dimension of A, $(n), does not match length of x, $(length(x))"
        ))
    end

    # Small case: don't use FFT
    N = m + n - 1
    if N < 512
        # Scale/initialize y
        if iszero(β)
            fill!(y, 0)
        else
            rmul!(y, β)
        end

        @inbounds for j in 1:n
            tmp = α * x[j]
            for i in 1:m
                y[i] = muladd(tmp, A[i,j], y[i])
            end
        end
    else
        # Large case: use FFT
        mul!(y, factorize(A), x, α, β)
    end

    return y
end
function mul!(
    y::StridedVector, A::ToeplitzFactorization, x::StridedVector, α::Number, β::Number
)
    n = length(x)
    m = length(y)
    vcvr_dft = A.vcvr_dft
    N = length(vcvr_dft)
    if m > N || n > N
        throw(DimensionMismatch(
            "Toeplitz factorization does not match size of input and output vector"
        ))
    end

    T = Base.promote_eltype(y, A, x, α, β)
    tmp = A.tmp
    dft = A.dft
    @inbounds begin
        copyto!(tmp, 1, x, 1, n)
        for i in (n+1):N
            tmp[i] = zero(eltype(tmp))
        end
        mul!(tmp, dft, tmp)
        for i in 1:N
            tmp[i] *= vcvr_dft[i]
        end
        dft \ tmp
        if iszero(β)
            for i in 1:m
                y[i] = α * maybereal(T, tmp[i])
            end
        else
            for i in 1:m
                y[i] = muladd(α, maybereal(T, tmp[i]), β * y[i])
            end
        end
    end

    return y
end

# Application of a general Toeplitz matrix to a general matrix
function mul!(
    C::StridedMatrix, A::AbstractToeplitz, B::StridedMatrix, α::Number, β::Number
)
    return mul!(C, factorize(A), B, α, β)
end
function mul!(
    C::StridedMatrix, A::ToeplitzFactorization, B::StridedMatrix, α::Number, β::Number
)
    l = size(B, 2)
    if size(C, 2) != l
        throw(DimensionMismatch("input and output matrices must have same number of columns"))
    end
    for j = 1:l
        mul!(view(C, :, j), A, view(B, :, j), α, β)
    end
    return C
end

# Left division of a general matrix B by a general Toeplitz matrix A, i.e. the solution x of Ax=B.
function ldiv!(A::AbstractToeplitz, B::StridedMatrix)
    if size(A, 1) != size(A, 2)
        error("Division: Rectangular case is not supported.")
    end
    for j = 1:size(B, 2)
        ldiv!(A, view(B, :, j))
    end
    return B
end

function (\)(A::AbstractToeplitz, b::AbstractVector)
    T = promote_type(eltype(A), eltype(b))
    if T != eltype(A)
        throw(ArgumentError("promotion of Toeplitz matrices not handled yet"))
    end
    bb = similar(b, T)
    copyto!(bb, b)
    ldiv!(A, bb)
end

function factorize(A::Toeplitz)
    T = eltype(A)
    m, n = size(A)
    S = promote_type(float(T), Complex{Float32})
    tmp = Vector{S}(undef, m + n - 1)
    copyto!(tmp, A.vc)
    copyto!(tmp, m + 1, Iterators.reverse(A.vr), 1, n - 1)
    dft = plan_fft!(tmp)
    return ToeplitzFactorization{T,typeof(A),S,typeof(dft)}(dft * tmp, similar(tmp), dft)
end

function ldiv!(A::Toeplitz, b::StridedVector)
    preconditioner = factorize(strang(A))
    copyto!(b, IterativeLinearSolvers.cgs(A, zeros(eltype(b), length(b)), b, preconditioner, 1000, 100eps())[1])
end

# SymmetricToeplitz

function factorize(A::SymmetricToeplitz{T}) where {T<:Number}
    vc = A.vc
    m = length(vc)
    S = promote_type(float(T), Complex{Float32})
    tmp = Vector{S}(undef, 2 * m)
    copyto!(tmp, vc)
    @inbounds tmp[m + 1] = zero(T)
    copyto!(tmp, m + 2, Iterators.reverse(vc), 1, m - 1)
    dft = plan_fft!(tmp)
    return ToeplitzFactorization{T,typeof(A),S,typeof(dft)}(dft * tmp, similar(tmp), dft)
end

ldiv!(A::SymmetricToeplitz, b::StridedVector) = copyto!(b, IterativeLinearSolvers.cg(A, zeros(length(b)), b, strang(A), 1000, 100eps())[1])

function cholesky!(L::AbstractMatrix, T::SymmetricToeplitz)

    L[:, 1] .= T.vc ./ sqrt(T.vc[1])
    v = copy(L[:, 1])
    N = size(T, 1)

    @inbounds for n in 1:N-1
        sinθn = v[n + 1] / L[n, n]
        cosθn = sqrt(1 - sinθn^2)

        for n′ in n+1:N
            v[n′] = (v[n′] - sinθn * L[n′ - 1, n]) / cosθn
            L[n′, n + 1] = -sinθn * v[n′] + cosθn * L[n′ - 1, n]
        end
    end
    return Cholesky(L, 'L', 0)
end

"""
    cholesky(T::SymmetricToeplitz)

Implementation of the Bareiss Algorithm, adapted from "On the stability of the Bareiss and
related Toeplitz factorization algorithms", Bojanczyk et al, 1993.
"""
function cholesky(T::SymmetricToeplitz)
    return cholesky!(Matrix{eltype(T)}(undef, size(T, 1), size(T, 1)), T)
end

# circulant
const CirculantFactorization{T, V<:AbstractVector{T}} = ToeplitzFactorization{T,Circulant{T,V}}
function factorize(C::Circulant)
    T = eltype(C)
    vc = C.vc
    S = promote_type(float(T), Complex{Float32})
    tmp = Vector{S}(undef, length(vc))
    copyto!(tmp, vc)
    dft = plan_fft!(tmp)
    return ToeplitzFactorization{T,typeof(C),S,typeof(dft)}(dft * tmp, similar(tmp), dft)
end

Base.:*(A::Circulant, B::Circulant) = factorize(A) * factorize(B)
Base.:*(A::CirculantFactorization, B::Circulant) = A * factorize(B)
Base.:*(A::Circulant, B::CirculantFactorization) = factorize(A) * B
function Base.:*(A::CirculantFactorization, B::CirculantFactorization)
    A_vcvr_dft = A.vcvr_dft
    B_vcvr_dft = B.vcvr_dft
    m = length(A_vcvr_dft)
    n = length(B_vcvr_dft)
    if m != n
        throw(DimensionMismatch(
            "size of matrix A, $(m)x$(m), does not match size of matrix B, $(n)x$(n)"
        ))
    end

    vc = A.dft \ (A_vcvr_dft .* B_vcvr_dft)

    return Circulant(maybereal(Base.promote_eltype(A, B), vc))
end

# Make an eager adjoint, similar to adjoints of Diagonal in LinearAlgebra
adjoint(C::CirculantFactorization{T,V,S,P}) where {T,V,S,P} =
    CirculantFactorization{T,V,S,P}(conj.(C.vcvr_dft), C.tmp, C.dft)
Base.:*(A::Adjoint{<:Any,<:Circulant}, B::Circulant) = factorize(parent(A))' * factorize(B)
Base.:*(A::Adjoint{<:Any,<:Circulant}, B::CirculantFactorization) =
    factorize(parent(A))' * B
Base.:*(A::Adjoint{<:Any,<:Circulant}, B::Adjoint{<:Any,<:Circulant}) =
    factorize(parent(A))' * factorize(parent(B))'
Base.:*(A::Circulant, B::Adjoint{<:Any,<:Circulant}) = factorize(A) * factorize(parent(B))'
Base.:*(A::CirculantFactorization, B::Adjoint{<:Any,<:Circulant}) = A * factorize(parent(B))'

ldiv!(C::Circulant, b::AbstractVector) = ldiv!(factorize(C), b)
function ldiv!(C::CirculantFactorization, b::AbstractVector)
    n = length(b)
    tmp = C.tmp
    vcvr_dft = C.vcvr_dft
    if !(length(tmp) == length(vcvr_dft) == n)
        throw(DimensionMismatch(
            "size of Toeplitz factorization does not match the length of the output vector"
        ))
    end
    dft = C.dft
    copyto!(tmp, b)
    dft * tmp
    tmp ./= vcvr_dft
    dft \ tmp
    T = eltype(C)
    b .= maybereal.(T, tmp)
    return b
end

function inv(C::Circulant)
    F = factorize(C)
    vdft = map(inv, F.vcvr_dft)
    vc = F.dft \ vdft
    return Circulant(maybereal(eltype(C), vc))
end
function inv(C::CirculantFactorization{T,V,S,P}) where {T,V,S,P}
    vdft = map(inv, C.vcvr_dft)
    return CirculantFactorization{T,V,S,P}(vdft, similar(vdft), C.dft)
end

function strang(A::AbstractMatrix{T}) where T
    n = size(A, 1)
    v = Vector{T}(undef, n)
    n2 = div(n, 2)
    for i = 1:n2 + 1
        v[i] = A[i,1]
    end
    for i in n2+2:n
        v[i] = A[1, n - i + 2]
    end
    return Circulant(v)
end
function chan(A::AbstractMatrix{T}) where T
    n = size(A, 1)
    v = Vector{T}(undef, n)
    for i = 1:n
        v[i] = ((n - i + 1) * A[i, 1] + (i - 1) * A[1, min(n - i + 2, n)]) / n
    end
    return Circulant(v)
end

function pinv(C::Circulant, tol::Real=eps(real(float(one(eltype(C))))))
    F = factorize(C)
    vdft = map(F.vcvr_dft) do x
        z = inv(x)
        return abs(x) < tol ? zero(z) : z
    end
    vc = F.dft \ vdft
    return Circulant(maybereal(eltype(C), vc))
end

eigvals(C::Circulant) = eigvals(factorize(C))
eigvals(C::CirculantFactorization) = copy(C.vcvr_dft)
_det(C) = prod(eigvals(C))
det(C::Circulant) = _det(C)
det(C::Circulant{<:Real}) = real(_det(C))
@static if VERSION <= v"1.6"
    _cispi(x) = cis(pi*x)
else
    _cispi(x) = cispi(x)
end
function eigvecs(C::Circulant)
    n = size(C,1)
    M = Array{complex(float(eltype(C)))}(undef, size(C))
    x = 2/n
    invnorm = 1/√n
    for CI in CartesianIndices(M)
        k, j = Tuple(CI)
        M[CI] = _cispi((k-1) * (j-1) * x) * invnorm
    end
    return M
end
function eigen(C::Circulant)
    Eigen(eigvals(C), eigvecs(C))
end

sqrt(C::Circulant) = sqrt(factorize(C))
function sqrt(C::CirculantFactorization)
    vc = C.dft \ sqrt.(C.vcvr_dft)
    return Circulant(maybereal(eltype(C), vc))
end

function _vc_first_rest_rev(C::Circulant)
    v = _vc(C)
    v1 = first(v)
    vrest = @view v[firstindex(v)+1:lastindex(v)]
    vrestrev = view(vrest, reverse(eachindex(vrest)))
    v1, vrest, vrestrev
end

function issymmetric(C::Circulant)
    v1, vrest, vrestrev = _vc_first_rest_rev(C)
    issymmetric(v1) && all(((a,b),) -> a == transpose(b), zip(vrest, vrestrev))
end
function ishermitian(C::Circulant)
    v1, vrest, vrestrev = _vc_first_rest_rev(C)
    ishermitian(v1) && all(((a,b),) -> a == adjoint(b), zip(vrest, vrestrev))
end

# Triangular

# NB! only valid for lower triangular
function smallinv(A::LowerTriangularToeplitz{T}) where T
    n = size(A, 1)
    b = zeros(T, n)
    b[1] = 1 ./ A.v[1]
    for k = 2:n
        tmp = zero(T)
        for i = 1:k-1
            tmp += A.v[k - i + 1]*b[i]
        end
        b[k] = -tmp/A.v[1]
    end
    return LowerTriangularToeplitz(b)
end
function inv(A::LowerTriangularToeplitz{T}) where T
    n = size(A, 1)
    if n <= 64
        return smallinv(A)
    end
    np2 = nextpow(2, n)
    if n != np2
        return LowerTriangularToeplitz(
            inv(LowerTriangularToeplitz([A.v; zeros(T, np2 - n)])).v[1:n])
    end
    nd2 = div(n, 2)
    a1 = inv(LowerTriangularToeplitz(A.v[1:nd2])).v
    return LowerTriangularToeplitz(
        [a1; -(LowerTriangularToeplitz(a1) * (Toeplitz(A.v[nd2 + 1:end], A.v[nd2 + 1:-1:2]) * a1))]
    )
end

function smallinv(A::UpperTriangularToeplitz{T}) where T
    n = size(A, 1)
    b = zeros(T, n)
    b[1] = 1 ./ A.v[1]
    for k = 2:n
        tmp = zero(T)
        for i = 1:k-1
            tmp += A.v[i + 1] * b[k - i]
        end
        b[k] = -tmp/A.v[1]
    end
    return UpperTriangularToeplitz(b)
end
function inv(A::UpperTriangularToeplitz{T}) where T
    n = size(A, 1)
    if n <= 64
        return smallinv(A)
    end
    np2 = nextpow(2, n)
    if n != np2
        return UpperTriangularToeplitz(
            inv(UpperTriangularToeplitz([A.v; zeros(T, np2 - n)])).v[1:n]
        )
    end
    nd2 = div(n, 2)
    a1 = inv(UpperTriangularToeplitz(A.v[1:nd2])).v
    return UpperTriangularToeplitz(
        [a1; -(UpperTriangularToeplitz(a1) * (Toeplitz(A.v[nd2 + 1:end], A.v[nd2 + 1:-1:2]) * a1))]
    )
end

# ldiv!(A::TriangularToeplitz,b::StridedVector) = inv(A)*b
function ldiv!(A::TriangularToeplitz, b::StridedVector)
    preconditioner = factorize(chan(A))
    copyto!(b, IterativeLinearSolvers.cgs(A, zeros(eltype(b), length(b)), b, preconditioner, 1000, 100eps())[1])
end

function (*)(A::TriangularToeplitz, B::TriangularToeplitz)
    n = size(A, 2)
    if n != size(B, 1)
        throw(DimensionMismatch(""))
    end
    if A.uplo == B.uplo
        return TriangularToeplitz(conv(A.v, B.v)[1:n], A.uplo)
    end
    return A * Matrix(B)
end

function factorize(A::LowerTriangularToeplitz)
    T = eltype(A)
    v = A.v
    n = length(v)
    S = promote_type(float(T), Complex{Float32})
    tmp = zeros(S, 2 * n - 1)
    copyto!(tmp, v)
    dft = plan_fft!(tmp)
    return ToeplitzFactorization{T,typeof(A),S,typeof(dft)}(dft * tmp, similar(tmp), dft)
end
function factorize(A::UpperTriangularToeplitz)
    T = eltype(A)
    v = A.v
    n = length(v)
    S = promote_type(float(T), Complex{Float32})
    tmp = zeros(S, 2 * n - 1)
    tmp[1] = v[1]
    copyto!(tmp, n + 1, Iterators.reverse(v), 1, n - 1)
    dft = plan_fft!(tmp)
    return ToeplitzFactorization{T,typeof(A),S,typeof(dft)}(dft * tmp, similar(tmp), dft)
end
