module ToeplitzMatrices
import DSP: conv

import Base: adjoint, convert, transpose, size, getindex, similar, copy, getproperty, inv, sqrt, copyto!, reverse, conj, zero, fill!, checkbounds, real, imag, isfinite, DimsInteger, iszero
import Base: parent
import Base: ==, +, -, *, \
import Base: AbstractMatrix
import LinearAlgebra: Cholesky, Factorization
import LinearAlgebra: ldiv!, factorize, lmul!, pinv, eigvals, eigvecs, eigen, Eigen, det
import LinearAlgebra: cholesky!, cholesky, tril!, triu!, checksquare, rmul!, dot, mul!, tril, triu, diag
import LinearAlgebra: istriu, istril, isdiag
import LinearAlgebra: UpperTriangular, LowerTriangular, Symmetric, Adjoint
import LinearAlgebra: issymmetric, ishermitian
import LinearAlgebra: eigvals, eigvecs, eigen

import AbstractFFTs: Plan, plan_fft!

using FillArrays
using LinearAlgebra
const AbstractFillVector{T} = FillArrays.AbstractFill{T,1}
const HermOrSym{T,M} = Union{Hermitian{T,M}, Symmetric{T,M}}

export AbstractToeplitz, Toeplitz, SymmetricToeplitz, Circulant, LowerTriangularToeplitz, UpperTriangularToeplitz, TriangularToeplitz, Hankel
export durbin, trench, levinson

@static if isdefined(Base, :require_one_based_indexing)
    const require_one_based_indexing = Base.require_one_based_indexing
else
    function require_one_based_indexing(A...)
        !Base.has_offset_axes(A...) || throw(ArgumentError("offset arrays are not supported but got an array with index other than 1"))
    end
end

include("iterativeLinearSolvers.jl")

# Abstract
abstract type AbstractToeplitz{T} <: AbstractMatrix{T} end

size(A::AbstractToeplitz) = (length(A.vc),length(A.vr))
@inline _vr(A::AbstractToeplitz) = A.vr
@inline _vc(A::AbstractToeplitz) = A.vc
@inline _vr(A::AbstractMatrix) = A[1,:]
@inline _vc(A::AbstractMatrix) = A[:,1]

AbstractArray{T}(A::AbstractToeplitz) where T = AbstractToeplitz{T}(A)
convert(::Type{AbstractToeplitz{T}}, A::AbstractToeplitz) where T = AbstractToeplitz{T}(A)

isconcrete(A::AbstractToeplitz) = isconcretetype(typeof(A.vc)) && isconcretetype(typeof(A.vr))
iszero(A::AbstractToeplitz) = iszero(A.vc) && iszero(A.vr)
function diag(A::AbstractToeplitz, k::Integer=0)
    if k >= size(A, 2) || -k >= size(A, 1)
        Fill(zero(eltype(A)), 0)
    elseif k >= 0
        Fill(A.vr[k + 1], diaglenpos(size(A)..., k))
    else
        Fill(A.vc[-k + 1], diaglenneg(size(A)..., k))
    end
end

function istril(A::AbstractToeplitz, k::Integer=0)
    vr, vc = _vr(A), _vc(A)
    all(iszero, @view vr[max(1, k+2):end]) && all(iszero, @view vc[2:min(-k,end)])
end

function istriu(A::AbstractToeplitz, k::Integer=0)
    vr, vc = _vr(A), _vc(A)
    all(iszero, @view vc[max(1, -k+2):end]) && all(iszero, @view vr[2:min(k,end)])
end

function isdiag(A::AbstractToeplitz)
    vr, vc = _vr(A), _vc(A)
    all(iszero, @view vr[2:end]) && all(iszero, @view vc[2:end])
end

function zero!(v::AbstractVector, inds = eachindex(v))
    if eltype(v) <: Number && isconcretetype(eltype(v))
        if inds == eachindex(v)
            v .= zero(eltype(v))
        else
            v[inds] .= zero(eltype(v))
        end
    else
        if inds == eachindex(v)
            v .= zero.(v)
        else
            @views v[inds] .= zero.(v[inds])
        end
    end
    return v
end

"""
    ToeplitzFactorization

Factorization of a Toeplitz matrix using FFT.
"""
struct ToeplitzFactorization{T,A<:AbstractToeplitz{T},S<:Number,P<:Plan{S}} <: Factorization{T}
    vcvr_dft::Vector{S}
    tmp::Vector{S}
    dft::P
end

include("toeplitz.jl")
include("special.jl")
include("hankel.jl")
include("linearalgebra.jl")
include("eigen.jl")

"""
    maybereal(::Type{T}, x)

Return real-valued part of `x` if `T` is a type of a real number, and `x` otherwise.
"""
maybereal(::Type, x) = x
maybereal(::Type{<:Real}, x) = real(x)

# length of LinearAlgebra.diagind, for positive and negative k respectively
@inline diaglenpos(m::Integer, n::Integer, k::Integer=0) = min(m, n-k)
@inline diaglenneg(m::Integer, n::Integer, k::Integer=0) = min(m+k, n)

include("directLinearSolvers.jl")

if !isdefined(Base, :get_extension)
    include("../ext/ToeplitzMatricesStatsBaseExt.jl")
end

end #module
