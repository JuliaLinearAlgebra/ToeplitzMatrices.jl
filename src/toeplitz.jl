# General Toeplitz matrix
"""
    Toeplitz

A Toeplitz matrix.
"""
struct Toeplitz{T, VC<:AbstractVector{T}, VR<:AbstractVector{T}} <: AbstractToeplitz{T}
    vc::VC
    vr::VR

    function Toeplitz{T, VC, VR}(vc::VC, vr::VR) where {T, VC<:AbstractVector{T}, VR<:AbstractVector{T}}
        require_one_based_indexing(vr, vc)
        if first(vc) != first(vr)
            error("First element of the vectors must be the same")
        end
        return new{T,VC,VR}(vc, vr)
    end
end
Toeplitz{T}(vc::AbstractVector{T}, vr::AbstractVector{T}) where T = Toeplitz{T,typeof(vc),typeof(vr)}(vc,vr)

"""
    Toeplitz(vc::AbstractVector, vr::AbstractVector)

Create a `Toeplitz` matrix from its first column `vc` and first row `vr` where
`vc[1] == vr[1]`.
"""
function Toeplitz(vc::AbstractVector, vr::AbstractVector)
    return Toeplitz{Base.promote_eltype(vc, vr)}(vc, vr)
end
function Toeplitz{T}(vc::AbstractVector, vr::AbstractVector) where T
    return Toeplitz{T}(convert(AbstractVector{T}, vc), convert(AbstractVector{T}, vr))
end

"""
    Toeplitz(A::AbstractMatrix)

"Project" matrix `A` onto its Toeplitz part using the first row/col of `A`.
"""
Toeplitz(A::AbstractMatrix) = Toeplitz{eltype(A)}(A)
Toeplitz{T}(A::AbstractMatrix) where {T} = Toeplitz{T}(copy(_vc(A)), copy(_vr(A)))

AbstractToeplitz{T}(A::Toeplitz) where T = Toeplitz{T}(A)
convert(::Type{Toeplitz{T}}, A::AbstractToeplitz) where {T} = A isa Toeplitz{T} ? A : Toeplitz{T}(A)
convert(::Type{Toeplitz}, A::AbstractToeplitz) = A isa Toeplitz ? A : Toeplitz(A)

# Retrieve an entry
Base.@propagate_inbounds function getindex(A::AbstractToeplitz, i::Integer, j::Integer)
    @boundscheck checkbounds(A,i,j)
    d = i - j
    if d >= 0
        return A.vc[d + 1]
    else
        return A.vr[1 - d]
    end
end

checknonaliased(A::Toeplitz) = Base.mightalias(A.vc, A.vr) && throw(ArgumentError("Cannot modify Toeplitz matrices in place with aliased data"))

function _copymutable(v::AbstractVector)
    w = similar(v)
    w .= v
    return w
end
_copymutable(A::Toeplitz) = Toeplitz(_copymutable(A.vc), _copymutable(A.vr))

function tril!(A::Toeplitz, k::Integer=0)
    checknonaliased(A)

    if k >= 0
        i1, iend = firstindex(A.vr), lastindex(A.vr)
        inds = max(k+2,i1):iend
        zero!(A.vr, inds)
    else
        i1, iend = firstindex(A.vc), lastindex(A.vc)
        inds = i1:min(-k,iend)
        zero!(A.vr)
        zero!(A.vc, inds)
    end
    A
end
function triu!(A::Toeplitz, k::Integer=0)
    checknonaliased(A)

    if k <= 0
        i1, iend = firstindex(A.vc), lastindex(A.vc)
        inds = max(-k+2,i1):iend
        zero!(A.vc, inds)
    else
        i1, iend = firstindex(A.vr), lastindex(A.vr)
        inds = i1:min(k,iend)
        zero!(A.vc)
        zero!(A.vr, inds)
    end
    A
end

tril(A::AbstractToeplitz, k::Integer=0) = tril!(convert(Toeplitz, _copymutable(A)), k)
triu(A::AbstractToeplitz, k::Integer=0) = triu!(convert(Toeplitz, _copymutable(A)), k)

adjoint(A::AbstractToeplitz) = transpose(conj(A))
transpose(A::AbstractToeplitz) = Toeplitz(A.vr, A.vc)
function AbstractMatrix{T}(A::AbstractToeplitz) where {T}
    vc = AbstractVector{T}(_vc(A))
    vr = AbstractVector{T}(_vr(A))
    Toeplitz{T}(vc,vr)
end
for fun in (:zero, :conj, :copy, :-, :real, :imag)
    @eval $fun(A::AbstractToeplitz)=Toeplitz($fun(A.vc),$fun(A.vr))
end
for op in (:+, :-)
    @eval $op(A::AbstractToeplitz,B::AbstractToeplitz)=Toeplitz($op(A.vc,B.vc),$op(A.vr,B.vr))
end
function copyto!(A::Toeplitz, B::AbstractToeplitz)
    checknonaliased(A)
    copyto!(A.vc,B.vc)
    copyto!(A.vr,B.vr)
    A
end
(==)(A::AbstractToeplitz,B::AbstractToeplitz) = A.vr==B.vr && A.vc==B.vc

function fill!(A::Toeplitz, x::Number)
    fill!(A.vc,x)
    fill!(A.vr,x)
    A
end
(*)(scalar::Number, C::AbstractToeplitz) = Toeplitz(scalar * C.vc, scalar * C.vr)
(*)(C::AbstractToeplitz,scalar::Number) = Toeplitz(C.vc * scalar, C.vr * scalar)

function lmul!(x::Number, A::Toeplitz)
    checknonaliased(A)
    lmul!(x, A.vc)
    lmul!(x, A.vr)
    A
end
function rmul!(A::Toeplitz, x::Number)
    checknonaliased(A)
    rmul!(A.vc, x)
    rmul!(A.vr, x)
    A
end
