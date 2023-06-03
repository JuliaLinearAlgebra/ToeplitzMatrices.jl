module ToeplitzMatrices
# import StatsBase: levinson!, levinson
import DSP: conv

import Base: adjoint, convert, transpose, size, getindex, similar, copy, getproperty, inv, sqrt, copyto!, reverse, conj, zero, fill!, checkbounds, real, imag, isfinite, DimsInteger, iszero
import Base: parent
import Base: ==, +, -, *, \
import Base: AbstractMatrix
import LinearAlgebra: Cholesky, Factorization
import LinearAlgebra: ldiv!, factorize, lmul!, pinv, eigvals, eigvecs, eigen, Eigen, det
import LinearAlgebra: cholesky!, cholesky, tril!, triu!, checksquare, rmul!, dot, mul!, tril, triu
import LinearAlgebra: UpperTriangular, LowerTriangular, Symmetric, Adjoint
import AbstractFFTs: Plan, plan_fft!
import StatsBase

export AbstractToeplitz, Toeplitz, SymmetricToeplitz, Circulant, LowerTriangularToeplitz, UpperTriangularToeplitz, TriangularToeplitz, Hankel
export durbin, trench, levinson

include("iterativeLinearSolvers.jl")

# Abstract
abstract type AbstractToeplitz{T<:Number} <: AbstractMatrix{T} end

size(A::AbstractToeplitz) = (length(A.vc),length(A.vr))
@inline _vr(A::AbstractToeplitz) = A.vr
@inline _vc(A::AbstractToeplitz) = A.vc
@inline _vr(A::AbstractMatrix) = A[1,:]
@inline _vc(A::AbstractMatrix) = A[:,1]

AbstractArray{T}(A::AbstractToeplitz) where T = AbstractToeplitz{T}(A)
convert(::Type{AbstractToeplitz{T}}, A::AbstractToeplitz) where T = AbstractToeplitz{T}(A)

isconcrete(A::AbstractToeplitz) = isconcretetype(typeof(A.vc)) && isconcretetype(typeof(A.vr))
iszero(A::AbstractToeplitz) = iszero(A.vc) && iszero(A.vr)

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

"""
    maybereal(::Type{T}, x)

Return real-valued part of `x` if `T` is a type of a real number, and `x` otherwise.
"""
maybereal(::Type, x) = x
maybereal(::Type{<:Real}, x) = real(x)

include("directLinearSolvers.jl")

end #module
