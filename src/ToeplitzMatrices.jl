module ToeplitzMatrices
using StatsBase


import Base: convert, *, \, getindex, print_matrix, size, Matrix, +, -, copy, similar,
    sqrt, copyto!, adjoint, transpose
import LinearAlgebra: cholesky, cholesky!, eigvals, inv, ldiv!, mul!, pinv, tril, triu

using LinearAlgebra: LinearAlgebra, Adjoint, Factorization, factorize, Cholesky,
    DimensionMismatch, rmul!

using AbstractFFTs
using AbstractFFTs: Plan

flipdim(A, d) = reverse(A, dims=d)

export Toeplitz, SymmetricToeplitz, Circulant, TriangularToeplitz, Hankel, chan, strang


include("iterativeLinearSolvers.jl")

# Abstract
abstract type AbstractToeplitz{T<:Number} <: AbstractMatrix{T} end

"""
    ToeplitzFactorization

Factorization of a Toeplitz matrix using FFT.
"""
struct ToeplitzFactorization{T<:Number,A<:AbstractToeplitz{T},S<:Number,P<:Plan{S}} <: Factorization{T}
    vcvr_dft::Vector{S}
    tmp::Vector{S}
    dft::P
end

size(A::AbstractToeplitz) = (size(A, 1), size(A, 2))
function getindex(A::AbstractToeplitz, i::Integer)
    return A[mod(i - 1, size(A, 1)) + 1, div(i - 1, size(A, 1)) + 1]
end

convert(::Type{AbstractMatrix{T}}, S::AbstractToeplitz) where {T} = convert(AbstractToeplitz{T}, S)
convert(::Type{AbstractArray{T}}, S::AbstractToeplitz) where {T} = convert(AbstractToeplitz{T}, S)

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

# General Toeplitz matrix
"""
    Toeplitz

A Toeplitz matrix.
"""
struct Toeplitz{T<:Number} <: AbstractToeplitz{T}
    vc::Vector{T}
    vr::Vector{T}

    function Toeplitz{T}(vc::Vector{T}, vr::Vector{T}) where {T<:Number}
        if first(vc) != first(vr)
            error("First element of the vectors must be the same")
        end
        return new{T}(vc, vr)
    end
end

"""
    Toeplitz(vc::AbstractVector, vr::AbstractVector)

Create a `Toeplitz` matrix from its first column `vc` and first row `vr` where
`vc[1] == vr[1]`.
"""
function Toeplitz(vc::AbstractVector, vr::AbstractVector)
    return Toeplitz{Base.promote_eltype(vc, vr)}(vc, vr)
end
function Toeplitz{T}(vc::AbstractVector, vr::AbstractVector) where {T<:Number}
    return Toeplitz{T}(convert(Vector{T}, vc), convert(Vector{T}, vr))
end

"""
    Toeplitz(A::AbstractMatrix)

"Project" matrix `A` onto its Toeplitz part using the first row/col of `A`.
"""
Toeplitz(A::AbstractMatrix) = Toeplitz{eltype(A)}(A)
function Toeplitz{T}(A::AbstractMatrix) where {T<:Number}
    return Toeplitz{T}(A[:,1], A[1,:])
end

function LinearAlgebra.factorize(A::Toeplitz)
    T = eltype(A)
    m, n = size(A)
    S = promote_type(T, Complex{Float32})
    tmp = Vector{S}(undef, m + n - 1)
    copyto!(tmp, A.vc)
    copyto!(tmp, m + 1, Iterators.reverse(A.vr), 1, n - 1)
    dft = plan_fft!(tmp)
    return ToeplitzFactorization{T,typeof(A),S,typeof(dft)}(dft * tmp, similar(tmp), dft)
end

convert(::Type{AbstractToeplitz{T}}, A::Toeplitz) where {T} = convert(Toeplitz{T}, A)
convert(::Type{Toeplitz{T}}, A::Toeplitz) where {T} = Toeplitz(convert(Vector{T}, A.vc),
                                                               convert(Vector{T}, A.vr))

adjoint(A::Toeplitz) = Toeplitz(conj(A.vr), conj(A.vc))
adjoint(A::Toeplitz{<:Real}) = transpose(A)
transpose(A::Toeplitz) = Toeplitz(A.vr, A.vc)

# Size of a general Toeplitz matrix
function size(A::Toeplitz, dim::Int)
    if dim == 1
        return length(A.vc)
    elseif dim == 2
        return length(A.vr)
    elseif dim > 2
        return 1
    else
        error("arraysize: dimension out of range")
    end
end

# Retrieve an entry
function getindex(A::Toeplitz, i::Integer, j::Integer)
    m, n  = size(A)
    @boundscheck if i < 1 || i > m || j < 1 || j > n
        throw(BoundsError(A, (i,j)))
    end
    d = i - j
    if d >= 0
        return A.vc[d + 1]
    else
        return A.vr[1 - d]
    end
end

# Form a lower triangular Toeplitz matrix by annihilating all entries above the k-th diaganal
function tril(A::Toeplitz, k = 0)
    if k > 0
        error("Second argument cannot be positive")
    end
    Al = TriangularToeplitz(copy(A.vc), 'L', length(A.vr))
    if k < 0
        for i in 1:-k
            Al.ve[i] = zero(eltype(A))
        end
    end
    return Al
end

# Form a lower triangular Toeplitz matrix by annihilating all entries below the k-th diagonal
function triu(A::Toeplitz, k = 0)
    if k < 0
        error("Second argument cannot be negative")
    end
    Al = TriangularToeplitz(copy(A.vr), 'U', length(A.vc))
    if k > 0
        for i in 1:k
            Al.ve[i] = zero(eltype(A))
        end
    end
    return Al
end

function ldiv!(A::Toeplitz, b::StridedVector)
    preconditioner = factorize(strang(A))
    copyto!(b, IterativeLinearSolvers.cgs(A, zeros(eltype(b), length(b)), b, preconditioner, 1000, 100eps())[1])
end

# Symmetric
"""
    SymmetricToeplitz

A symmetric Toeplitz matrix.
"""
struct SymmetricToeplitz{T<:Number} <: AbstractToeplitz{T}
    vc::Vector{T}
end

SymmetricToeplitz(vc::AbstractVector) = SymmetricToeplitz{eltype(vc)}(vc)

SymmetricToeplitz(A::AbstractMatrix) = SymmetricToeplitz{eltype(A)}(A)
SymmetricToeplitz{T}(A::AbstractMatrix) where {T<:Number} = SymmetricToeplitz{T}(A[1, :])

function LinearAlgebra.factorize(A::SymmetricToeplitz{T}) where {T<:Number}
    vc = A.vc
    m = length(vc)
    S = promote_type(T, Complex{Float32})
    tmp = Vector{S}(undef, 2 * m)
    copyto!(tmp, vc)
    @inbounds tmp[m + 1] = zero(T)
    copyto!(tmp, m + 2, Iterators.reverse(vc), 1, m - 1)
    dft = plan_fft!(tmp)
    return ToeplitzFactorization{T,typeof(A),S,typeof(dft)}(dft * tmp, similar(tmp), dft)
end

convert(::Type{AbstractToeplitz{T}}, A::SymmetricToeplitz) where {T} = convert(SymmetricToeplitz{T},A)
convert(::Type{SymmetricToeplitz{T}}, A::SymmetricToeplitz) where {T} = SymmetricToeplitz(convert(Vector{T},A.vc))

adjoint(A::SymmetricToeplitz) = SymmetricToeplitz(conj(A.vr), conj(A.vc))
adjoint(A::SymmetricToeplitz{<:Real}) = A
transpose(A::SymmetricToeplitz) = A

function size(A::SymmetricToeplitz, dim::Int)
    if dim < 1
        error("arraysize: dimension out of range")
    end
    return dim < 3 ? length(A.vc) : 1
end

function getindex(A::SymmetricToeplitz, i::Integer, j::Integer)
    m  = size(A, 1)
    @boundscheck if i < 1 || i > m || j < 1 || j > m
        throw(BoundsError(A, (i,j)))
    end
    return A.vc[abs(i - j) + 1]
end

ldiv!(A::SymmetricToeplitz, b::StridedVector) =
    copyto!(b, IterativeLinearSolvers.cg(A, zeros(length(b)), b, strang(A), 1000, 100eps())[1])

# Circulant
"""
    Circulant

A circulant matrix.
"""
struct Circulant{T<:Number} <: AbstractToeplitz{T}
    vc::Vector{T}
end

"""
    Circulant(vc::AbstractVector{<:Number})

Create a circulant matrix from its first column `vc`.
"""
Circulant(vc::AbstractVector) = Circulant{eltype(vc)}(vc)

"""
    Circulant(A::AbstractMatrix)

Create a circulant matrix from the first column of matrix `A`.
"""
Circulant(A::AbstractMatrix) = Circulant{eltype(A)}(A)
Circulant{T}(A::AbstractMatrix) where {T<:Number} = Circulant{T}(A[:,1])

const CirculantFactorization{T<:Number} = ToeplitzFactorization{T,Circulant{T}}
function LinearAlgebra.factorize(C::Circulant)
    T = eltype(C)
    vc = C.vc
    S = promote_type(T, Complex{Float32})
    tmp = Vector{S}(undef, length(vc))
    copyto!(tmp, vc)
    dft = plan_fft!(tmp)
    return ToeplitzFactorization{T,typeof(C),S,typeof(dft)}(dft * tmp, similar(tmp), dft)
end

convert(::Type{AbstractToeplitz{T}}, A::Circulant) where {T} = convert(Circulant{T}, A)
convert(::Type{Circulant{T}}, A::Circulant) where {T} = Circulant(convert(Vector{T}, A.vc))


function size(C::Circulant, dim::Integer)
    if dim < 1
        error("arraysize: dimension out of range")
    end
    return dim < 3 ? length(C.vc) : 1
end

function getindex(C::Circulant, i::Integer, j::Integer)
    n = size(C, 1)
    @boundscheck if i < 1 || i > n || j < 1 || j > n
        throw(BoundsError(C, (i,j)))
    end
    return C.vc[mod(i - j, length(C.vc)) + 1]
end

LinearAlgebra.ldiv!(C::Circulant, b::AbstractVector) = ldiv!(factorize(C), b)
function LinearAlgebra.ldiv!(C::CirculantFactorization, b::AbstractVector)
    n = length(b)
    tmp = C.tmp
    vcvr_dft = C.vcvr_dft
    if !(length(tmp) == length(vcvr_dft) == n)
        throw(DimensionMismatch(
            "size of Toeplitz factorization does not match the length of the output vector"
        ))
    end
    dft = C.dft
    @inbounds begin
        copyto!(tmp, b)
        dft * tmp
        tmp ./= vcvr_dft
        dft \ tmp
        T = eltype(C)
        b .= maybereal.(T, tmp)
    end
    return b
end

function Base.inv(C::Circulant)
    F = factorize(C)
    vdft = map(inv, F.vcvr_dft)
    vc = F.dft \ vdft
    return Circulant(maybereal(eltype(C), vc))
end
function Base.inv(C::CirculantFactorization)
    vdft = map(inv, C.vcvr_dft)
    return CirculantFactorization(vdft, similar(vdft), C.dft)
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

function LinearAlgebra.pinv(C::Circulant, tol::Real=eps(real(float(one(eltype(C))))))
    F = factorize(C)
    vdft = map(F.vcvr_dft) do x
        z = inv(x)
        return abs(x) < tol ? zero(z) : z
    end
    vc = F.dft \ vdft
    return Circulant(maybereal(eltype(C), vc))
end

LinearAlgebra.eigvals(C::Circulant) = eigvals(factorize(C))
LinearAlgebra.eigvals(C::CirculantFactorization) = copy(C.vcvr_dft)

sqrt(C::Circulant) = sqrt(factorize(C))
function Base.sqrt(C::CirculantFactorization)
    vc = C.dft \ sqrt.(C.vcvr_dft)
    return Circulant(maybereal(eltype(C), vc))
end

copy(C::Circulant) = Circulant(copy(C.vc))
similar(C::Circulant) = Circulant(similar(C.vc))
function copyto!(dest::Circulant, src::Circulant)
    copyto!(dest.vc, src.vc)
    return dest
end

(+)(C1::Circulant, C2::Circulant) = Circulant(C1.vc + C2.vc)
(-)(C1::Circulant, C2::Circulant) = Circulant(C1.vc - C2.vc)
(-)(C::Circulant) = Circulant(-C.vc)

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

Base.:*(A::Adjoint{<:Circulant}, B::Circulant) = factorize(parent(A))' * factorize(B)
Base.:*(A::Adjoint{<:Circulant}, B::CirculantFactorization) = factorize(parent(A))' * B
Base.:*(A::Adjoint{<:CirculantFactorization}, B::Circulant) = A * factorize(B)
function Base.:*(A::Adjoint{<:CirculantFactorization}, B::CirculantFactorization)
    C = parent(A)
    C_vcvr_dft = C.vcvr_dft
    B_vcvr_dft = B.vcvr_dft
    m = length(C_vcvr_dft)
    n = length(B_vcvr_dft)
    if m != n
        throw(DimensionMismatch(
            "size of matrix A, $(m)x$(m), does not match size of matrix B, $(n)x$(n)"
        ))
    end

    tmp = conj.(C_vcvr_dft) .* B_vcvr_dft
    vc = C.dft \ tmp

    return Circulant(maybereal(Base.promote_eltype(C, B), vc))
end

(*)(scalar::Number, C::Circulant) = Circulant(scalar * C.vc)
(*)(C::Circulant,scalar::Number) = Circulant(C.vc * scalar)


# Triangular
struct TriangularToeplitz{T<:Number} <: AbstractToeplitz{T}
    ve::Vector{T}
    uplo::Char
end

function TriangularToeplitz(ve::AbstractVector, uplo::Symbol)
    return TriangularToeplitz{eltype(ve)}(ve, uplo)
end
function TriangularToeplitz{T}(ve::AbstractVector, uplo::Symbol) where {T<:Number}
    UL = LinearAlgebra.char_uplo(uplo)
    return TriangularToeplitz{T}(convert(Vector{T}, ve), UL)
end

function TriangularToeplitz(A::AbstractMatrix, uplo::Symbol)
    return TriangularToeplitz{eltype(A)}(A, uplo)
end
function TriangularToeplitz{T}(A::AbstractMatrix, uplo::Symbol) where {T<:Number}
    ve = uplo === :U ? A[1, :] : A[:, 1]
    return TriangularToeplitz{T}(ve, uplo)
end

function LinearAlgebra.factorize(A::TriangularToeplitz)
    T = eltype(A)
    ve = A.ve
    n = length(ve)
    S = promote_type(T, Complex{Float32})
    tmp = zeros(S, 2 * n - 1)
    if A.uplo === 'L'
        copyto!(tmp, ve)
    else
        tmp[1] = ve[1]
        copyto!(tmp, n + 1, Iterators.reverse(ve), 1, n - 1)
    end
    dft = plan_fft!(tmp)
    return ToeplitzFactorization{T,typeof(A),S,typeof(dft)}(dft * tmp, similar(tmp), dft)
end

function convert(::Type{Toeplitz}, A::TriangularToeplitz)
    if A.uplo == 'L'
        Toeplitz(A.ve, [A.ve[1]; zeros(length(A.ve) - 1)])
    else
        @assert A.uplo == 'U'
        Toeplitz([A.ve[1]; zeros(length(A.ve) - 1)], A.ve)
    end
end

convert(::Type{AbstractToeplitz{T}}, A::TriangularToeplitz) where {T} = convert(TriangularToeplitz{T},A)
convert(::Type{TriangularToeplitz{T}}, A::TriangularToeplitz) where {T} =
    TriangularToeplitz(convert(Vector{T},A.ve),A.uplo=='U' ? (:U) : (:L))


function size(A::TriangularToeplitz, dim::Int)
    if dim < 1
        error("arraysize: dimension out of range")
    end
    return dim < 3 ? length(A.ve) : 1
end

function getindex(A::TriangularToeplitz{T}, i::Integer, j::Integer) where T
    n = size(A, 1)
    @boundscheck if i < 1 || i > n || j < 1 || j > n
        throw(BoundsError(A, (i,j)))
    end
    if A.uplo == 'L'
        return i >= j ? A.ve[i - j + 1] : zero(T)
    else
        return i <= j ? A.ve[j - i + 1] : zero(T)
    end
end

function (*)(A::TriangularToeplitz, B::TriangularToeplitz)
    n = size(A, 1)
    if n != size(B, 1)
        throw(DimensionMismatch(""))
    end
    if A.uplo == B.uplo
        return TriangularToeplitz(conv(A.ve, B.ve)[1:n], A.uplo)
    end
    return Triangular(Matrix(A), A.uplo) * Triangular(Matrix(B), B.uplo)
end

function Base.:*(A::Adjoint{<:TriangularToeplitz}, b::AbstractVector)
    M = parent(A)
    return TriangularToeplitz{eltype(M)}(M.ve, M.uplo) * b
end

# NB! only valid for lower triangular
function smallinv(A::TriangularToeplitz{T}) where T
    n = size(A, 1)
    b = zeros(T, n)
    b[1] = 1 ./ A.ve[1]
    for k = 2:n
        tmp = zero(T)
        for i = 1:k-1
            tmp += A.uplo == 'L' ? A.ve[k - i + 1]*b[i] : A.ve[i + 1] * b[k - i]
        end
        b[k] = -tmp/A.ve[1]
    end
    return TriangularToeplitz(b, symbol(A.uplo))
end

function inv(A::TriangularToeplitz{T}) where T
    n = size(A, 1)
    if n <= 64
        return smallinv(A)
    end
    np2 = nextpow2(n)
    if n != np2
        return TriangularToeplitz(inv(TriangularToeplitz([A.ve, zeros(T, np2 - n)],
            symbol(A.uplo))).ve[1:n], symbol(A.uplo))
    end
    nd2 = div(n, 2)
    a1 = inv(TriangularToeplitz(A.ve[1:nd2], symbol(A.uplo))).ve
    return TriangularToeplitz([a1, -(TriangularToeplitz(a1, symbol(A.uplo)) *
        (Toeplitz(A.ve[nd2 + 1:end], A.ve[nd2 + 1:-1:2]) * a1))], symbol(A.uplo))
end

# ldiv!(A::TriangularToeplitz,b::StridedVector) = inv(A)*b
function ldiv!(A::TriangularToeplitz, b::StridedVector)
    preconditioner = factorize(chan(A))
    copyto!(b, IterativeLinearSolvers.cgs(A, zeros(eltype(b), length(b)), b, preconditioner, 1000, 100eps())[1])
end

# extend levinson
StatsBase.levinson!(x::StridedVector, A::SymmetricToeplitz, b::StridedVector) =
    StatsBase.levinson!(A.vc, b, x)
function StatsBase.levinson!(C::StridedMatrix, A::SymmetricToeplitz, B::StridedMatrix)
    n = size(B, 2)
    if n != size(C, 2)
        throw(DimensionMismatch("input and output matrices must have same number of columns"))
    end
    for j = 1:n
        StatsBase.levinson!(view(C, :, j), A, view(B, :, j))
    end
    C
end
StatsBase.levinson(A::AbstractToeplitz, B::StridedVecOrMat) =
    StatsBase.levinson!(zeros(size(B)), A, copy(B))

# BlockTriangular
# type BlockTriangularToeplitz{T<:BlasReal} <: AbstractMatrix{T}
#     Mc::Array{T,3}
#     uplo::Char
#     Mc_dft::Array{Complex{T},3}
#     tmp::Vector{Complex{T}}
#     dft::Plan
# end
# function BlockTriangularToeplitz{T<:BlasReal}(Mc::Array{T,3}, uplo::Symbol)
#     n, p, _ = size(Mc)
#     tmp = zeros(Complex{T}, 2n)
#     dft = plan_fft!(tmp)
#     Mc_dft = Array{Complex{T}}(2n, p, p)
#     for j = 1:p
#         for i = 1:p
#             Mc_dft[1,i,j] = complex(Mc[1,i,j])
#             for t = 2:n
#                 Mc_dft[t,i,j] = uplo == :L ? complex(Mc[t,i,j]) : zero(Complex{T})
#             end
#             Mc_dft[n+1,i,j] = zero(Complex{T})
#             for t = n+2:2n
#                 Mc_dft[t,i,j] = uplo == :L ? zero(Complex{T}) : complex(Mc[2n-t+2,i,j])
#             end
#             dft(view(Mc_dft, 1:2n, 1:p, 1:p))
#         end
#     end
#     return BlockTriangularToeplitz(Mc, string(uplo)[1], Mc_dft, tmp, dft, idft)
# end

#= Hankel Matrix
 A Hankel matrix is a matrix that is constant across the anti-diagonals:

  [a_0 a_1 a_2 a_3 a_4
   a_1 a_2 a_3 a_4 a_5
   a_2 a_3 a_4 a_5 a_6]

 This is precisely a Toeplitz matrix with the columns reversed:
                             [0 0 0 0 1
  [a_4 a_3 a_2 a_1 a_0        0 0 0 1 0
   a_5 a_4 a_3 a_2 a_1   *    0 0 1 0 0
   a_6 a_5 a_4 a_3 a_2]       0 1 0 0 0
                              1 0 0 0 0]
 We represent the Hankel matrix by wrapping the corresponding Toeplitz matrix.=#


# Hankel Matrix, use _Hankel as Hankel(::Toeplitz) should project to Hankel
function _Hankel end

# Hankel Matrix
mutable struct Hankel{TT<:Number} <: AbstractMatrix{TT}
    T::Toeplitz{TT}
    global _Hankel(T::Toeplitz{TT}) where TT<:Number = new{TT}(T)
end

# Ctor: vc is the leftmost column and vr is the bottom row.
function Hankel{T}(vc::AbstractVector, vr::AbstractVector) where T
    if vc[end] != vr[1]
        error("First element of rows must equal last element of columns")
    end
    n = length(vr)
    p = [vc; vr[2:end]]
    _Hankel(Toeplitz{T}(p[n:end],p[n:-1:1]))
end

Hankel(vc::AbstractVector, vr::AbstractVector) =
    Hankel{promote_type(eltype(vc), eltype(vr))}(vc, vr)

Hankel{T}(A::AbstractMatrix) where T = Hankel{T}(A[:,1], A[end,:])
Hankel(A::AbstractMatrix) = Hankel(A[:,1], A[end,:])

convert(::Type{AbstractArray{T}}, A::Hankel) where {T<:Number} = convert(Hankel{T}, A)
convert(::Type{AbstractMatrix{T}}, A::Hankel) where {T<:Number} = convert(Hankel{T}, A)
convert(::Type{Hankel{T}}, A::Hankel) where {T<:Number} = _Hankel(convert(Toeplitz{T}, A.T))



# Size
size(H::Hankel, k...) = size(H.T, k...)


# Retrieve an entry by two indices
function getindex(A::Hankel, i::Integer, j::Integer)
    m, n = size(A)
    @boundscheck if i < 1 || i > m || j < 1 || j > n
        throw(BoundsError(A, (i,j)))
    end
    return A.T[i,n-j+1]
end

# Fast application of a general Hankel matrix to a general vector
*(A::Hankel, b::AbstractVector) = A.T * reverse(b)

# Fast application of a general Hankel matrix to a general matrix
*(A::Hankel, B::AbstractMatrix) = A.T * flipdim(B, 1)
## BigFloat support

(*)(A::Toeplitz{T}, b::AbstractVector) where {T<:BigFloat} = irfft(
    rfft([
        A.vc;
        reverse(A.vr[2:end])]
    ) .* rfft([
        b;
        zeros(length(b) - 1)
    ]),
    2 * length(b) - 1
)[1:length(b)]

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

Implementation of the Bareiss Algorhithm, adapted from "On the stability of the Bareiss and
related Toeplitz factorization algorithms", Bojanczyk et al, 1993.
"""
function cholesky(T::SymmetricToeplitz)
    return cholesky!(Matrix{eltype(T)}(undef, size(T, 1), size(T, 1)), T)
end

"""
    maybereal(::Type{T}, x)

Return real-valued part of `x` if `T` is a type of a real number, and `x` otherwise.
"""
maybereal(::Type, x) = x
maybereal(::Type{<:Real}, x) = real(x)

end #module
