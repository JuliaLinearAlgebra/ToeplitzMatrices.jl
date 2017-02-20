__precompile__(true)

module ToeplitzMatrices
    using Compat


import StatsBase
include("iterativeLinearSolvers.jl")

import Base: *, \, full, getindex, print_matrix, size, tril, triu, inv, A_mul_B!, Ac_mul_B,
    A_ldiv_B!, convert
import Base.LinAlg: BlasReal, DimensionMismatch


export Toeplitz, SymmetricToeplitz, Circulant, TriangularToeplitz, Hankel,
       chan, strang

# Abstract
@compat abstract type AbstractToeplitz{T<:Number} <: AbstractMatrix{T} end

size(A::AbstractToeplitz) = (size(A, 1), size(A, 2))
getindex(A::AbstractToeplitz, i::Integer) = A[mod(i, size(A,1)), div(i, size(A,1)) + 1]


convert(::Type{Matrix}, S::AbstractToeplitz) = full(S)
convert{T}(::Type{AbstractMatrix{T}}, S::AbstractToeplitz) = convert(AbstractToeplitz{T}, S)
convert{T}(::Type{AbstractArray{T}}, S::AbstractToeplitz) = convert(AbstractToeplitz{T}, S)


# Convert an abstract Toeplitz matrix to a full matrix
function full{T}(A::AbstractToeplitz{T})
    m, n = size(A)
    Af = Matrix{T}(m, n)
    for j = 1:n
        for i = 1:m
            Af[i,j] = A[i,j]
        end
    end
    return Af
end

# Fast application of a general Toeplitz matrix to a column vector via FFT
function A_mul_B!{T}(α::T, A::AbstractToeplitz{T}, x::StridedVector{T}, β::T,
      y::StridedVector{T})
    m = size(A,1)
    n = size(A,2)
    N = length(A.vcvr_dft)
    if m != length(y)
        throw(DimensionMismatch(""))
    end
    if n != length(x)
        throw(DimensionMismatch(""))
    end
    if N < 512
        y[:] *= β
        for j = 1:n
            tmp = α * x[j]
            for i = 1:m
                y[i] += tmp*A[i,j]
            end
        end
        return y
    end
    for i = 1:n
        A.tmp[i] = x[i]
    end
    for i = n+1:N
        A.tmp[i] = 0
    end
    A_mul_B!(A.tmp, A.dft, A.tmp)
    for i = 1:N
        A.tmp[i] *= A.vcvr_dft[i]
    end
    A.dft \ A.tmp
    for i = 1:m
        y[i] *= β
        y[i] += α * (T <: Real ? real(A.tmp[i]) : A.tmp[i])
    end
    return y
end

# Application of a general Toeplitz matrix to a general matrix
function A_mul_B!{T}(α::T, A::AbstractToeplitz{T}, B::StridedMatrix{T}, β::T,
    C::StridedMatrix{T})
    l = size(B, 2)
    if size(C, 2) != l
        throw(DimensionMismatch("input and output matrices must have same number of columns"))
    end
    for j = 1:l
        A_mul_B!(α, A, view(B, :, j), β, view(C, :, j))
    end
    return C
end

# Translate three to five argument A_mul_B!
A_mul_B!(y, A::AbstractToeplitz, x) = A_mul_B!(one(eltype(A)), A, x, zero(eltype(A)), x)

# * operator
function (*)(A::AbstractToeplitz, B::StridedVecOrMat)
    T = promote_type(eltype(A), eltype(B))
    A_mul_B!(one(T), convert(AbstractMatrix{T}, A), convert(AbstractArray{T}, B), zero(T),
        size(B,2) == 1 ? zeros(T, size(A, 1)) : zeros(T, size(A, 1), size(B, 2)))
end

# Left division of a general matrix B by a general Toeplitz matrix A, i.e. the solution x of Ax=B.
function A_ldiv_B!(A::AbstractToeplitz, B::StridedMatrix)
    if size(A, 1) != size(A, 2)
        error("Division: Rectangular case is not supported.")
    end
    for j = 1:size(B, 2)
        A_ldiv_B!(A, view(B, :, j))
    end
    return B
end

function (\)(A::AbstractToeplitz, b::AbstractVector)
    T = promote_type(eltype(A), eltype(b))
    if T != eltype(A)
        throw(ArgumentError("promotion of Toeplitz matrices not handled yet"))
    end
    bb = similar(b, T)
    copy!(bb, b)
    A_ldiv_B!(A, bb)
end

# General Toeplitz matrix
type Toeplitz{T<:Number,S<:Number} <: AbstractToeplitz{T}
    vc::Vector{T}
    vr::Vector{T}
    vcvr_dft::Vector{S}
    tmp::Vector{S}
    dft::Base.DFT.Plan{S}
end

# Ctor
function Toeplitz(vc::Vector, vr::Vector)
    m, n = length(vc), length(vr)
    if vc[1] != vr[1]
        error("First element of the vectors must be the same")
    end

    T = promote_type(eltype(vc), eltype(vr), Float32)
    vcp, vrp = Vector{T}(vc), Vector{T}(vr)

    tmp = Vector{promote_type(T, Complex{Float32})}(m + n - 1)
    copy!(tmp, vcp)
    for i = 1:n - 1
        tmp[i + m] = vrp[n - i + 1]
    end
    dft = plan_fft!(tmp)
    return Toeplitz(vcp, vrp, dft*tmp, similar(tmp), dft)
end

convert{T}(::Type{AbstractToeplitz{T}},A::Toeplitz) = convert(Toeplitz{T},A)
convert{T}(::Type{Toeplitz{T}},A::Toeplitz) = Toeplitz(convert(Vector{T},A.vc),
                                                       convert(Vector{T},A.vr))

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
    m = size(A,1)
    n = size(A,2)
    if i > m || j > n
        error("BoundsError()")
    end

    if i >= j
        return A.vc[i - j + 1]
    else
        return A.vr[1 - i + j]
    end
end

# Form a lower triangular Toeplitz matrix by annihilating all entries above the k-th diaganal
function tril(A::Toeplitz, k = 0)
    if k > 0
        error("Second argument cannot be positive")
    end
    Al = TriangularToeplitz(copy(A.vc), 'L', length(A.vr))
    if k < 0
      for i in -1:-1:k
          Al.ve[-i] = zero(eltype(A))
      end
    end
    return Al
end

# Form a lower triangular Toeplitz matrix by annihilating all entries below the k-th diaganal
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

A_ldiv_B!(A::Toeplitz, b::StridedVector) =
    copy!(b, IterativeLinearSolvers.cgs(A, zeros(eltype(b), length(b)), b, strang(A), 1000, 100eps())[1])

# Symmetric
type SymmetricToeplitz{T<:BlasReal} <: AbstractToeplitz{T}
    vc::Vector{T}
    vcvr_dft::Vector{Complex{T}}
    tmp::Vector{Complex{T}}
    dft::Base.DFT.Plan
end
function SymmetricToeplitz{T<:BlasReal}(vc::Vector{T})
    tmp = convert(Array{Complex{T}}, [vc; zero(T); reverse(vc[2:end])])
    dft = plan_fft!(tmp)
    return SymmetricToeplitz(vc, dft*tmp, similar(tmp), dft)
end

convert{T}(::Type{AbstractToeplitz{T}},A::SymmetricToeplitz) = convert(SymmetricToeplitz{T},A)
convert{T}(::Type{SymmetricToeplitz{T}},A::SymmetricToeplitz) = SymmetricToeplitz(convert(Vector{T},A.vc))

function size(A::SymmetricToeplitz, dim::Int)
    if 1 <= dim <= 2
        return length(A.vc)
    else
        error("arraysize: dimension out of range")
    end
end

getindex(A::SymmetricToeplitz, i::Integer, j::Integer) = A.vc[abs(i - j) + 1]

A_ldiv_B!(A::SymmetricToeplitz, b::StridedVector) =
    copy!(b, IterativeLinearSolvers.cg(A, zeros(length(b)), b, strang(A), 1000, 100eps())[1])

# Circulant
type Circulant{T<:Number,S<:Number} <: AbstractToeplitz{T}
    vc::Vector{T}
    vcvr_dft::Vector{S}
    tmp::Vector{S}
    dft::Base.DFT.Plan
end

function Circulant{T<:Real}(vc::Vector{T})
    TT = promote_type(eltype(vc), Float32)
    tmp = zeros(promote_type(TT, Complex{Float32}), length(vc))
    return Circulant(real(vc), fft(vc), tmp, plan_fft!(tmp))
end
function Circulant(vc::Vector)
    T = promote_type(eltype(vc), Float32)
    tmp = zeros(promote_type(T, Complex{Float32}), length(vc))
    return Circulant(vc, fft(vc), tmp, plan_fft!(tmp))
end

convert{T}(::Type{AbstractToeplitz{T}},A::Circulant) = convert(Circulant{T},A)
convert{T}(::Type{Circulant{T}},A::Circulant) = Circulant(convert(Vector{T},A.vc))


function size(C::Circulant, dim::Integer)
    if 1 <= dim <= 2
        return length(C.vc)
    else
        error("arraysize: dimension out of range")
    end
end

function getindex(C::Circulant, i::Integer, j::Integer)
    n = size(C, 1)
    if i > n || j > n
        error("BoundsError()")
    end
    return C.vc[mod(i - j, length(C.vc)) + 1]
end

function Ac_mul_B{T<:Real}(A::Circulant{T}, B::Circulant{T})
    tmp = similar(A.vcvr_dft)
    for i = 1:length(tmp)
        tmp[i] = conj(A.vcvr_dft[i]) * B.vcvr_dft[i]
    end
    return Circulant(real(A.vc)(A.dft \ tmp), tmp, A.tmp, A.dft)
end
function Ac_mul_B(A::Circulant, B::Circulant)
    T = promote_type(eltype(A), eltype(B))
    tmp = similar(A.vcvr_dft, T)
    for i = 1:length(tmp)
        tmp[i] = conj(A.vcvr_dft[i]) * B.vcvr_dft[i]
    end
    tmp2 = A.dft \ tmp
    return Circulant(Vector{T}(tmp2), tmp, Vector{T}(A.tmp), eltype(A) == T ? A.dft : plan_fft!(tmp2))
end

function A_ldiv_B!{T}(C::Circulant{T}, b::AbstractVector{T})
    n = length(b)
    size(C, 1) == n || throw(DimensionMismatch(""))
    for i = 1:n
        C.tmp[i] = b[i]
    end
    C.dft * C.tmp
    for i = 1:n
        C.tmp[i] /= C.vcvr_dft[i]
    end
    C.dft \ C.tmp
    for i = 1:n
        b[i] = (T <: Real ? real(C.tmp[i]) : C.tmp[i])
    end
    return b
end

function inv{T<:Real}(C::Circulant{T})
    vdft = 1 ./ C.vcvr_dft
    return Circulant(real(C.dft \ vdft), copy(vdft), similar(vdft), C.dft)
end
function inv(C::Circulant)
    vdft = 1 ./ C.vcvr_dft
    return Circulant(C.dft \ vdft, copy(vdft), similar(vdft), C.dft)
end

function strang{T}(A::AbstractMatrix{T})
    n = size(A, 1)
    v = Vector{T}(n)
    n2 = div(n, 2)
    for i = 1:n
        if i <= n2 + 1
            v[i] = A[i,1]
        else
            v[i] = A[1, n - i + 2]
        end
    end
    return Circulant(v)
end
function chan{T}(A::AbstractMatrix{T})
    n = size(A, 1)
    v = Vector{T}(n)
    for i = 1:n
        v[i] = ((n - i + 1) * A[i, 1] + (i - 1) * A[1, min(n - i + 2, n)]) / n
    end
    return Circulant(v)
end

# Triangular
type TriangularToeplitz{T<:Number,S<:Number} <: AbstractToeplitz{T}
    ve::Vector{T}
    uplo::Char
    vcvr_dft::Vector{S}
    tmp::Vector{S}
    dft::Base.DFT.Plan
end

function TriangularToeplitz(ve::Vector, uplo::Symbol)
    n = length(ve)

    T = promote_type(eltype(ve), Float32)
    vep = Vector{T}(ve)

    tmp = zeros(promote_type(T, Complex{Float32}), 2n - 1)
    if uplo == :L
        copy!(tmp, vep)
    else
        tmp[1] = vep[1]
        for i = 1:n - 1
            tmp[n + i] = ve[n - i + 1]
        end
    end
    dft = plan_fft!(tmp)
    return TriangularToeplitz(vep, string(uplo)[1], dft * tmp, similar(tmp), dft)
end

function convert(::Type{Toeplitz}, A::TriangularToeplitz)
    if A.uplo == 'L'
        Toeplitz(A.ve, [A.ve[1]; zeros(length(A.ve) - 1)])
    else
        @assert A.uplo == 'U'
        Toeplitz([A.ve[1]; zeros(length(A.ve) - 1)], A.ve)
    end
end

convert{T}(::Type{AbstractToeplitz{T}},A::TriangularToeplitz) = convert(TriangularToeplitz{T},A)
convert{T}(::Type{TriangularToeplitz{T}},A::TriangularToeplitz) =
    TriangularToeplitz(convert(Vector{T},A.ve),A.uplo=='U'?(:U):(:L))


function size(A::TriangularToeplitz, dim::Int)
    if dim == 1 || dim == 2
        return length(A.ve)
    elseif dim > 2
        return 1
    else
        error("arraysize: dimension out of range")
    end
end

function getindex{T}(A::TriangularToeplitz{T}, i::Integer, j::Integer)
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
    return Triangular(full(A), A.uplo) * Triangular(full(B), B.uplo)
end

Ac_mul_B(A::TriangularToeplitz, b::AbstractVector) =
    TriangularToeplitz(A.ve, A.uplo == 'U' ? :L : :U) * b

# NB! only valid for lower triangular
function smallinv{T}(A::TriangularToeplitz{T})
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

function inv{T}(A::TriangularToeplitz{T})
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

# A_ldiv_B!(A::TriangularToeplitz,b::StridedVector) = inv(A)*b
A_ldiv_B!(A::TriangularToeplitz, b::StridedVector) =
    copy!(b, IterativeLinearSolvers.cgs(A, zeros(eltype(b), length(b)), b, chan(A), 1000, 100eps())[1])

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
#     dft::Base.DFT.Plan
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

# Hankel Matrix
type Hankel{T<:Number} <: AbstractMatrix{T}
    T::Toeplitz{T}
end

# Ctor: vc is the leftmost column and vr is the bottom row.
function Hankel(vc,vr)
    if vc[end] != vr[1]
        error("First element of rows must equal last element of columns")
    end
    n = length(vr)
    p = [vc; vr[2:end]]
    Hankel(Toeplitz(p[n:end],p[n:-1:1]))
end

convert(::Type{Array},A::Hankel) = convert(Matrix,A)
convert(::Type{Matrix},A::Hankel) = full(A)

convert{T}(::Type{AbstractMatrix{T}},A::Hankel) = convert(Hankel{T},A)
convert{T}(::Type{Hankel{T}},A::Hankel) = Hankel(convert(Toeplitz{T},A.T))



# Size
size(H::Hankel,k...) = size(H.T,k...)

# Full version of a Hankel matrix
function full{T}(A::Hankel{T})
    m, n = size(A)
    Af = Matrix{T}(m, n)
    for j = 1:n
        for i = 1:m
            Af[i,j] = A[i,j]
        end
    end
    return Af
end

# Retrieve an entry by two indices
getindex(A::Hankel, i::Integer, j::Integer) = A.T[i,end-j+1]

# Retrieve an entry by one index
getindex(H::Hankel, i::Integer) = H[mod(i, size(H,1)), div(i, size(H,1)) + 1]

# Fast application of a general Hankel matrix to a general vector
*(A::Hankel,b::AbstractVector) = A.T * reverse(b)

# Fast application of a general Hankel matrix to a general matrix
*(A::Hankel,B::AbstractMatrix) = A.T * flipdim(B, 1)
## BigFloat support

(*){T<:BigFloat}(A::Toeplitz{T}, b::Vector) = irfft(
    rfft([
        A.vc;
        reverse(A.vr[2:end])]
    ) .* rfft([
        b;
        zeros(length(b) - 1)
    ]),
    2 * length(b) - 1
)[1:length(b)]

end #module
