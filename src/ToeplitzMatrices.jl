module ToeplitzMatrices
import StatsBase: levinson!, levinson
import DSP: conv

import Base: adjoint, convert, transpose, size, getindex, similar, copy, getproperty, inv, sqrt, copyto!, reverse, conj, zero, fill!, checkbounds, real, imag
import Base: ==, +, -, *, \
import LinearAlgebra: Cholesky, Factorization
import LinearAlgebra: ldiv!, factorize, lmul!, pinv, eigvals, Adjoint, cholesky!, cholesky, tril!, triu!
import LinearAlgebra: UpperTriangular, LowerTriangular, Symmetric
import AbstractFFTs: Plan, plan_fft!
import Infinities: ℵ₀

export AbstractToeplitz, Toeplitz, SymmetricToeplitz, Circulant, LowerTriangularToeplitz, UpperTriangularToeplitz, TriangularToeplitz, Hankel

include("iterativeLinearSolvers.jl")

# Abstract
abstract type AbstractToeplitz{T<:Number} <: AbstractMatrix{T} end

size(A::AbstractToeplitz) = (size(A, 1), size(A, 2))
@inline _vr(A::AbstractToeplitz) = A.vr
@inline _vc(A::AbstractToeplitz) = A.vc
@inline _vr(A::AbstractMatrix) = A[1,:]
@inline _vc(A::AbstractMatrix) = A[:,1]

convert(::Type{AbstractMatrix{T}}, S::AbstractToeplitz) where {T<:Number} = convert(AbstractToeplitz{T}, S)
convert(::Type{AbstractArray{T}}, S::AbstractToeplitz) where {T<:Number} = convert(AbstractToeplitz{T}, S)

isconcrete(A::AbstractToeplitz) = isconcretetype(typeof(A.vc)) && isconcretetype(typeof(A.vr))

"""
    ToeplitzFactorization

Factorization of a Toeplitz matrix using FFT.
"""
struct ToeplitzFactorization{T<:Number,A<:AbstractToeplitz{T},S<:Number,P<:Plan{S}} <: Factorization{T}
    vcvr_dft::Vector{S}
    tmp::Vector{S}
    dft::P
end

include("toeplitz.jl")
include("special.jl")
include("linearalgebra.jl")

"""
    maybereal(::Type{T}, x)

Return real-valued part of `x` if `T` is a type of a real number, and `x` otherwise.
"""
maybereal(::Type, x) = x
maybereal(::Type{<:Real}, x) = real(x)

end #module
