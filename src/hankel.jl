# Hankel
struct Hankel{T, V<:AbstractVector{T}, S<:DimsInteger} <: AbstractMatrix{T}
    v::V
    size::S # size

    function Hankel{T,V,S}(v::V, (m,n)::DimsInteger) where {T, V<:AbstractVector{T}, S<:DimsInteger}
        (m < 0 || n < 0) && throw(ArgumentError("negative size: $s"))
        require_one_based_indexing(v)
        length(v) ≥ m+n-1 || throw(ArgumentError("inconsistency between size and number of anti-diagonals"))
        new{T,V,S}(v, (m,n))
    end
end
Hankel{T}(v::V, s::S) where {T, V<:AbstractVector{T}, S<:DimsInteger} = Hankel{T,V,S}(v,s)

Hankel{T}(v::AbstractVector, s::DimsInteger) where T = Hankel{T}(convert(AbstractVector{T},v),s)
Hankel{T}(v::AbstractVector, h::Integer, w::Integer) where T = Hankel{T}(v,(h,w))
Hankel(v::AbstractVector, s::DimsInteger) = Hankel{eltype(v)}(v,s)
Hankel(v::AbstractVector, h::Integer, w::Integer) = Hankel{eltype(v)}(v,h,w)
Hankel(v::AbstractVector) = (l=length(v);Hankel(v,((l+1)÷2,(l+1)÷2))) # square by default
function Hankel(vc::AbstractVector, vr::AbstractVector)
    if vc[end] != vr[1]
        throw(ArgumentError("vc[end] != vr[1]"))
    end
    Hankel(vcat(vc,vr[2:end]), (length(vc), length(vr)))
end

function getproperty(A::Hankel, s::Symbol)
    m,_ = getfield(A, :size)
    if s == :vc
        A.v[1:m]
    elseif s == :vr
        A.v[m:end]
    else
        getfield(A,s)
    end
end

# convert from general matrix to Hankel matrix
Hankel{T}(A::AbstractMatrix, uplo::Symbol = :L) where T<:Number = convert(Hankel{T}, Hankel(A,uplo))
# using the first column and last row
function Hankel(A::AbstractMatrix, uplo::Symbol = :L)
    m,n = size(A)
    if uplo == :L
        if isfinite(m) # InfiniteArrays.jl supports infinite
            Hankel(A[:,1], A[end,:])
        else
            Hankel(A[:,1], (m,n))
        end
    elseif uplo == :U
        if isfinite(n)
            Hankel(A[1,:], A[:,end])
        else
            Hankel(A[1,:], (m,n))
        end
    else
        throw(ArgumentError("expected :L or :U. got $uplo."))
    end
end

convert(::Type{AbstractArray{T}}, A::Hankel) where {T<:Number} = convert(Hankel{T}, A)
convert(::Type{AbstractMatrix{T}}, A::Hankel) where {T<:Number} = convert(Hankel{T}, A)
convert(::Type{Hankel{T}}, A::Hankel) where {T<:Number} = Hankel{T}(convert(AbstractVector{T}, A.v), size(A))

# Size
size(H::Hankel) = H.size
diag(H::Hankel, k::Integer=0) = H.v[hankeldiagind(size(H)..., k)]
@inline hankeldiagind(m::Integer, n::Integer, k::Integer=0) = k ≥ 0 ? range(1 + k, step = 2, length = diaglenpos(m, n, k)) : range(1 - k, step = 2, length = diaglenneg(m, n, k))

# Retrieve an entry by two indices
Base.@propagate_inbounds function getindex(A::Hankel, i::Integer, j::Integer)
    @boundscheck checkbounds(A, i, j)
    return A.v[i+j-1]
end
AbstractMatrix{T}(A::Hankel) where {T} = Hankel{T}(AbstractVector{T}(A.v), A.size)
for fun in (:zero, :conj, :copy, :-, :similar, :real, :imag)
    @eval $fun(A::Hankel) = Hankel($fun(A.v), size(A))
end
for op in (:+, :-)
    @eval function $op(A::Hankel, B::Hankel)
        promote_shape(A,B)
        Hankel($op(A.v,B.v), size(A))
    end
end
function copyto!(A::Hankel, B::Hankel)
    promote_shape(A,B)
    copyto!(A.v, B.v)
    return A
end
for fun in (:lmul!,)
    @eval function $fun(x::Number, A::Hankel)
        $fun(x, A.v)
        A
    end
end
for fun in (:fill!, :rmul!)
    @eval function $fun(A::Hankel, x::Number)
        $fun(A.v, x)
        A
    end
end

transpose(A::Hankel) = Hankel(A.v, reverse(size(A)))
adjoint(A::Hankel) = transpose(conj(A))
(==)(A::Hankel, B::Hankel) = A.v == B.v && size(A) == size(B)
(*)(scalar::Number, C::Hankel) = Hankel(scalar * C.v, size(C))
(*)(C::Hankel,scalar::Number) = Hankel(C.v * scalar, size(C))

isconcrete(A::Hankel) = isconcretetype(A.v)

function reverse(A::Hankel; dims=1)
    _,n = size(A)
    if dims==1
        Toeplitz(reverse(A.vc), A.vr)
    elseif dims==2
        Toeplitz(A.v[n:end], A.v[n:-1:1])
    else
        throw(ArgumentError("invalid dimension $dims in reverse"))
    end
end
function reverse(A::AbstractToeplitz; dims=1)
    if dims==1
        Hankel(reverse(A.vc), A.vr)
    elseif dims==2
        Hankel(vcat(reverse(A.vr), A.vc), size(A))
    else
        throw(ArgumentError("invalid dimension $dims in reverse"))
    end
end

# Fast application of a general Hankel matrix to a strided vector
*(A::Hankel, b::StridedVector) = reverse(A, dims=2) * reverse(b)
mul!(y::StridedVector, A::Hankel, x::StridedVector, α::Number, β::Number) = mul!(y, reverse(A,dims=2), view(x, reverse(axes(x, 1))), α, β)
# Fast application of a general Hankel matrix to a strided matrix
*(A::Hankel, B::StridedMatrix) = reverse(A,dims=2) * reverse(B, dims=1)
mul!(Y::StridedMatrix, A::Hankel, X::StridedMatrix, α::Number, β::Number) = mul!(Y, reverse(A,dims=2), view(X, reverse(axes(X, 1)), :), α, β)
