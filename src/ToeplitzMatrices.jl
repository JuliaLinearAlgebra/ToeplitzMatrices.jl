module ToeplitzMatrices

import StatsBase
using IterativeLinearSolvers

import Base: *, \, full, getindex, print_matrix, size, tril, triu, inv, A_mul_B!, Ac_mul_B,
    A_ldiv_B!, convert
import Base.LinAlg: BlasReal, DimensionMismatch


export Toeplitz, SymmetricToeplitz, Circulant, TriangularToeplitz, Hankel,
       chan, strang


# Abstract
abstract AbstractToeplitz{T<:Number} <: AbstractMatrix{T}

size(A::AbstractToeplitz) = (size(A, 1), size(A, 2))
getindex(A::AbstractToeplitz, i::Integer) = A[mod(i, size(A,1)), div(i, size(A,1)) + 1]
function full{T}(A::AbstractToeplitz{T})
    m, n = size(A)
    Af = Array(T, m, n)
    for j = 1:n
        for i = 1:m
            Af[i,j] = A[i,j]
        end
    end
    return Af
end

function A_mul_B!{T}(α::T, A::AbstractToeplitz{T}, x::StridedVector{T}, β::T,
        y::StridedVector{T})
    n = length(y)
    n2 = length(A.vc_dft)
    if n != size(A,1)
        throw(DimensionMismatch(""))
    end
    if n != length(x)
        throw(DimensionMismatch(""))
    end
    if n < 512
        y[:] *= β
        for j = 1:n
            tmp = α * x[j]
            for i = 1:n
                y[i] += tmp*A[i,j]
            end
        end
        return y
    end
    for i = 1:n
        A.tmp[i] = complex(x[i])
    end
    for i = n+1:n2
        A.tmp[i] = zero(Complex{T})
    end
    A_mul_B!(A.tmp, A.dft, A.tmp)
    for i = 1:n2
        A.tmp[i] *= A.vc_dft[i]
    end
    A.dft \ A.tmp
    for i = 1:n
        y[i] *= β
        y[i] += α * real(A.tmp[i])
    end
    return y
end
function A_mul_B!{T}(α::T, A::AbstractToeplitz{T}, B::StridedMatrix{T}, β::T,
    C::StridedMatrix{T})
    n = size(B, 2)
    if size(C, 2) != n
        throw(DimensionMismatch("input and output matrices must have same number of columns"))
    end
    for j = 1:n
        A_mul_B!(α, A, sub(B, :, j), β, sub(C, :, j))
    end
    return C
end

(*){T}(A::AbstractToeplitz{T}, B::StridedVecOrMat{T}) =
    A_mul_B!(one(T), A, B, zero(T), zeros(T, size(B)))

function A_ldiv_B!(A::AbstractToeplitz, B::StridedMatrix)
    for j = 1:size(B, 2)
        A_ldiv_B!(A, sub(B, :, j))
    end
    return B
end

# General
type Toeplitz{T<:Number} <: AbstractToeplitz{T}
    vc::Vector{T}
    vr::Vector{T}
    vc_dft::Vector{Complex{T}}
    tmp::Vector{Complex{T}}
    dft::Base.DFT.Plan{Complex{T}}
end
function Toeplitz{T<:Number}(vc::Vector{T}, vr::Vector{T})
    n = length(vc)
    if length(vr) != n
        throw(DimensionMismatch(""))
    end
    if vc[1] != vr[1]
        error("First element of the vectors must be the same")
    end

    tmp = complex([vc; zero(T); reverse(vr[2:end])])
    dft = plan_fft!(tmp)
    return Toeplitz(vc, vr, dft*tmp, similar(tmp), dft)
end

Toeplitz{T<:Integer}(vc::Vector{T}, vr::Vector{T}) =
            Toeplitz(Vector{Float64}(vc),Vector{Float64}(vr))
Toeplitz{T<:Integer}(vc::Vector{Complex{T}}, vr::Vector{Complex{T}}) =
            Toeplitz(Vector{Complex128}(vc),Vector{Complex128}(vr))

function Toeplitz{T1<:Number,T2<:Number}(vc::Vector{T1},vr::Vector{T2})
    T=promote_type(T1,T2)
    Toeplitz(Vector{T}(vc),Vector{T}(vr))
end

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

function getindex(A::Toeplitz, i::Integer, j::Integer)
    n = size(A,1)
    if i > n || j > n
        error("BoundsError()")
    end

    if i >= j
        return A.vc[i - j + 1]
    else
        return A.vr[1 - i + j]
    end
end

function tril{T}(A::Toeplitz{T}, k::Integer)
    if k > 0
        error("Second argument cannot be positive")
    end
    Al = TriangularToeplitz(copy(A.vc), 'L')
    for i in -1:-1:k
        Al.ve[-i] = zero(T)
    end
    return Al
end
function triu{T}(A::Toeplitz{T}, k::Integer)
    if k < 0
        error("Second argument cannot be negative")
    end
    Al = TriangularToeplitz(copy(A.vr), 'U')
    for i in 1:k
        Al.ve[i] = zero(T)
    end
    return Al
end

A_ldiv_B!(A::Toeplitz, b::StridedVector) =
    copy!(b, cgs(A, zeros(length(b)), b, strang(A), 1000, 100eps())[1])

# Symmetric
type SymmetricToeplitz{T<:BlasReal} <: AbstractToeplitz{T}
    vc::Vector{T}
    vc_dft::Vector{Complex{T}}
    tmp::Vector{Complex{T}}
    dft::Base.DFT.Plan
end
function SymmetricToeplitz{T<:BlasReal}(vc::Vector{T})
    tmp = convert(Array{Complex{T}}, [vc; zero(T); reverse(vc[2:end])])
    dft = plan_fft!(tmp)
    return SymmetricToeplitz(vc, dft*tmp, similar(tmp), dft)
end

function size(A::SymmetricToeplitz, dim::Int)
    if 1 <= dim <= 2
        return length(A.vc)
    else
        error("arraysize: dimension out of range")
    end
end

getindex(A::SymmetricToeplitz, i::Integer, j::Integer) = A.vc[abs(i - j) + 1]

A_ldiv_B!(A::SymmetricToeplitz, b::StridedVector) =
    copy!(b, cg(A, zeros(length(b)), b, strang(A), 1000, 100eps())[1])

# Circulant
type Circulant{T<:BlasReal} <: AbstractToeplitz{T}
    vc::Vector{T}
    vc_dft::Vector{Complex{T}}
    tmp::Vector{Complex{T}}
    dft::Base.DFT.Plan
end
function Circulant{T<:BlasReal}(vc::Vector{T})
    tmp = zeros(Complex{T}, length(vc))
    return Circulant(vc, fft(vc), tmp, plan_fft!(tmp))
end

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

function Ac_mul_B(A::Circulant,B::Circulant)
    tmp = similar(A.vc_dft)
    for i = 1:length(tmp)
        tmp[i] = conj(A.vc_dft[i]) * B.vc_dft[i]
    end
    return Circulant(real(A.dft \ tmp), tmp, A.tmp, A.dft)
end

function A_ldiv_B!{T}(C::Circulant{T}, b::AbstractVector{T})
    n = length(b)
    size(C, 1) == n || throw(DimensionMismatch(""))
    for i = 1:n
        C.tmp[i] = b[i]
    end
    C.dft * C.tmp
    for i = 1:n
        C.tmp[i] /= C.vc_dft[i]
    end
    C.dft \ C.tmp
    for i = 1:n
        b[i] = real(C.tmp[i])
    end
    return b
end
(\)(C::Circulant, b::AbstractVector) = A_ldiv_B!(C, copy(b))

function inv{T<:BlasReal}(C::Circulant{T})
    vdft = 1 ./ C.vc_dft
    return Circulant(real(C.dft \ vdft), copy(vdft), similar(vdft), C.dft)
end

function strang{T}(A::AbstractMatrix{T})
    n = size(A, 1)
    v = Array(T, n)
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
    v = Array(T, n)
    for i = 1:n
        v[i] = ((n - i + 1) * A[i, 1] + (i - 1) * A[1, min(n - i + 2, n)]) / n
    end
    return Circulant(v)
end

# Triangular
type TriangularToeplitz{T<:Number} <: AbstractToeplitz{T}
    ve::Vector{T}
    uplo::Char
    vc_dft::Vector{Complex{T}}
    tmp::Vector{Complex{T}}
    dft::Base.DFT.Plan
end

function Toeplitz(A::TriangularToeplitz)
    if A.uplo == 'L'
        Toeplitz(A.ve,[A.ve[1];zeros(length(A.ve)-1)])
    else
        @assert A.uplo == 'U'
        Toeplitz([A.ve[1];zeros(length(A.ve)-1)],A.ve)
    end
end

function TriangularToeplitz{T<:Number}(ve::Vector{T}, uplo::Symbol)
    n = length(ve)
    tmp = uplo == :L ? complex([ve; zeros(n)]) : complex([ve[1]; zeros(T, n); reverse(ve[2:end])])
    dft = plan_fft!(tmp)
    return TriangularToeplitz(ve, string(uplo)[1], dft * tmp, similar(tmp), dft)
end

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
    copy!(b, cgs(A, zeros(length(b)), b, chan(A), 1000, 100eps())[1])

# extend levinson
StatsBase.levinson!(x::StridedVector, A::SymmetricToeplitz, b::StridedVector) =
    StatsBase.levinson!(A.vc, b, x)
function StatsBase.levinson!(C::StridedMatrix, A::SymmetricToeplitz, B::StridedMatrix)
    n = size(B, 2)
    if n != size(C, 2)
        throw(DimensionMismatch("input and output matrices must have same number of columns"))
    end
    for j = 1:n
        StatsBase.levinson!(sub(C, :, j), A, sub(B, :, j))
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
#     Mc_dft = Array(Complex{T}, 2n, p, p)
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
#             dft(sub(Mc_dft, 1:2n, 1:p, 1:p))
#         end
#     end
#     return BlockTriangularToeplitz(Mc, string(uplo)[1], Mc_dft, tmp, dft, idft)
# end

## Hankel
#  A hankel matrix has the form
#
#   [a_0 a_1 a_2;
#    a_1 a_2 a_3;
#    a_2 a_3 a_4]
#
#  This is precisely a Toeplitz matrix with the columns changed:
#
#   [a_2 a_1 a_0;
#    a_3 a_2 a_1;
#    a_4 a_3 a_2] * [0 0 1; 0 1 0; 1 0 0]
#
#  We represent the Hankel matrix by wrapping the corresponding Toeplitz matrix
#

type Hankel{T<:Number} <: AbstractMatrix{T}
    T::Toeplitz{T}
end


function Hankel(vc,vr)
    if vc[end] != vr[1]
        error("First element of rows must equal first element of columns")
    end    
    Hankel(Toeplitz(vr,reverse(vc)))
end

size(H::Hankel,k...) = size(H.T,k...)

getindex(H::Hankel, i::Integer) = H[mod(i, size(H,1)), div(i, size(H,1)) + 1]

function full{T}(A::Hankel{T})
    m, n = size(A)
    Af = Array(T, m, n)
    for j = 1:n
        for i = 1:m
            Af[i,j] = A[i,j]
        end
    end
    return Af
end

getindex(A::Hankel, i::Integer, j::Integer) = A.T[i,end-j+1]

*(A::Hankel,b::AbstractVector) = A.T*reverse(b)




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
