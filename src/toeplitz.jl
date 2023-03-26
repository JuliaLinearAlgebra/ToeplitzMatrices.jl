# General Toeplitz matrix
"""
    Toeplitz

A Toeplitz matrix.
"""
struct Toeplitz{T, VC<:AbstractVector{T}, VR<:AbstractVector{T}} <: AbstractToeplitz{T}
    vc::VC
    vr::VR

    function Toeplitz{T, VC, VR}(vc::VC, vr::VR) where {T, VC<:AbstractVector{T}, VR<:AbstractVector{T}}
        if !isequal(first(vc), first(vr))
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
convert(::Type{Toeplitz{T}}, A::AbstractToeplitz) where {T} = Toeplitz{T}(A)
convert(::Type{Toeplitz}, A::AbstractToeplitz) = Toeplitz(A)

# Dims for disambiguity
for DIM in (:DimsInteger, :Dims)
    @eval function similar(::Type{Toeplitz{T, VC, VR}}, dims::$DIM{2}) where {T, VC, VR}
        vc = similar(VC, dims[1])
        vr = similar(VR, dims[2])
        vr[1] = vc[1]
        Toeplitz(vc, vr)
    end
    @eval function similar(A::AbstractToeplitz, T::Type = eltype(A), dims::$DIM{2} = size(A))
        vc = similar(A.vc, T, dims[1])
        vr = similar(A.vr, T, dims[2])
        vr[1] = vc[1]
        Toeplitz{T}(vc, vr)
    end
end

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

broadcasted(::DefaultMatrixStyle, f, A::AbstractToeplitz) = Toeplitz(f.(A.vc), f.(A.vr))
broadcasted(::DefaultMatrixStyle, f, x::Number, A::AbstractToeplitz) = Toeplitz(f.(x, A.vc), f.(x, A.vr))
broadcasted(::DefaultMatrixStyle, f, A::AbstractToeplitz, x::Number) = Toeplitz(f.(A.vc, x), f.(A.vr, x))

checknonaliased(A::Toeplitz) = Base.mightalias(A.vc, A.vr) && throw(ArgumentError("Cannot modify Toeplitz matrices in place with aliased data"))

function tril!(A::Toeplitz, k::Integer=0)
    checknonaliased(A)

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
            A.vc=vcat(zero(A.vc[1:-k]), A.vc[-k+1:end])
        end
    end
    A
end
function triu!(A::Toeplitz, k::Integer=0)
    checknonaliased(A)

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
            A.vr=vcat(zero(A.vr[1:k]), A.vr[k+1:end])
        end
    end
    A
end

adjoint(A::AbstractToeplitz) = transpose(conj(A))
transpose(A::AbstractToeplitz) = Toeplitz(A.vr, A.vc)
for fun in (:zero, :copy)
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