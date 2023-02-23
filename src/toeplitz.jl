# General Toeplitz matrix
"""
    Toeplitz

A Toeplitz matrix.
"""
struct Toeplitz{T<:Number} <: AbstractToeplitz{T}
    vc::AbstractVector{T}
    vr::AbstractVector{T}

    function Toeplitz{T}(vc::AbstractVector{T}, vr::AbstractVector{T}) where {T<:Number}
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
    return Toeplitz{T}(convert(AbstractVector{T}, vc), convert(AbstractVector{T}, vr))
end

"""
    Toeplitz(A::AbstractMatrix)

"Project" matrix `A` onto its Toeplitz part using the first row/col of `A`.
"""
Toeplitz(A::AbstractMatrix) = Toeplitz{eltype(A)}(A)
Toeplitz{T}(A::AbstractMatrix) where {T<:Number} = Toeplitz{T}(A[:,1], A[1,:])

convert(::Type{AbstractToeplitz{T}}, A::Toeplitz) where {T} = convert(Toeplitz{T}, A)
convert(::Type{Toeplitz{T}}, A::Toeplitz) where {T} = Toeplitz(convert(AbstractVector{T}, A.vc),convert(AbstractVector{T}, A.vr))
convert(::Type{Toeplitz{T}}, A::AbstractToeplitz) where {T} = Toeplitz{T}(A.vc,A.vr)
convert(::Type{Toeplitz}, A::AbstractToeplitz) = Toeplitz(A.vc,A.vr)

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
    @boundscheck checkbounds(A,i,j)
    d = i - j
    if d >= 0
        return A.vc[d + 1]
    else
        return A.vr[1 - d]
    end
end

function tril!(A::Toeplitz, k::Integer=0)
    if k >= 0
        if isconcretetype(typeof(A.vr))
            for i in k+2:lastindex(A.vr)
                A.vr[i] = zero(eltype(A))
            end
        else
            A.vr=vcat(A.vr[1:k+1], zero(A.vr[k+2:end]))
        end
    else
        fill!(A.vr, zero(eltype(A)))
        if isconcretetype(typeof(A.vc))
            for i in 1:-k
                A.vc[i]=zero(eltype(A))
            end
        else
            A.vc=vcat(A.vc[1:-k+1], zero(A.vc[-k+2:end]))
        end
    end
    A
end
function triu!(A::Toeplitz, k::Integer=0)
    if k <= 0
        if isconcretetype(typeof(A.vc))
            for i in -k+2:lastindex(A.vc)
                A.vc[i] = zero(eltype(A))
            end
        else
            A.vc=vcat(A.vc[1:-k+1], zero(A.vc[-k+2:end]))
        end
    else
        fill!(A.vc, zero(eltype(A)))
        if isconcretetype(typeof(A.vr))
            for i in 1:k
                A.vr[i]=zero(eltype(A))
            end
        else
            A.vr=vcat(A.vr[1:k+1], zero(A.vr[k+2:end]))
        end
    end
    A
end

adjoint(A::AbstractToeplitz) = transpose(conj(A))
transpose(A::AbstractToeplitz) = Toeplitz(A.vr, A.vc)
for fun in (:zero, :conj, :copy, :-, :similar, :real, :imag)
    @eval begin
        $fun(A::AbstractToeplitz)=Toeplitz($fun(A.vc),$fun(A.vr))
    end
end
for op in (:+, :-)
    @eval begin
        $op(A::AbstractToeplitz,B::AbstractToeplitz)=Toeplitz($op(A.vc,B.vc),$op(A.vr,B.vr))
    end
end
function copyto!(A::Toeplitz, B::Toeplitz)
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