# Hankel
struct Hankel{T<:Number} <: AbstractMatrix{T}
    v::AbstractVector{T}
    s::Dims{2} # size
end

Hankel{T}(v::AbstractVector, h::Integer, w::Integer) where T = Hankel{T}(v,(h,w))
Hankel(v::AbstractVector, h::Integer, w::Integer) = Hankel{eltype(v)}(v,h,w)
Hankel(v::AbstractVector) = Hankel(v,((l+1)÷2,(l+1)÷2)) # square by default
function Hankel(vc::AbstractVector, vr::AbstractVector)
    if vc[end] != vr[1]
        throw(ArgumentError("vc[end] != vr[1]"))
    end
    Hankel(vcat(vc,vr[2:end]), (length(vc), length(vr)))
end

function getproperty(A::Hankel, s::Symbol)
    if s==:vc
        A.v[1:A.s[1]]
    elseif s==:vr
        A.v[A.s[1]:end]
    else
        getfield(A,s)
    end
end

# convert from general matrix to Hankel matrix
Hankel{T}(A::AbstractMatrix, uplo::Symbol = :L) where T<:Number = convert(Hankel{T}, Hankel(A,uplo))
# using the first column and last row
function Hankel(A::AbstractMatrix, uplo::Symbol = :L)
    s=size(A)
    if uplo == :L
        if isfinite(s[1])
            Hankel(A[:,1],A[end,:])
        else
            Hankel(A[:,1],s)
        end
    elseif uplo == :U
        if isfinite(s[2])
            Hankel(A[1,:],A[:,end])
        else
            Hankel(A[1,:],s)
        end
    else
        throw(ArgumentError("expected :L or :U. got $uplo."))
    end
end

convert(::Type{AbstractArray{T}}, A::Hankel) where {T<:Number} = convert(Hankel{T}, A)
convert(::Type{AbstractMatrix{T}}, A::Hankel) where {T<:Number} = convert(Hankel{T}, A)
convert(::Type{Hankel{T}}, A::Hankel) where {T<:Number} = Hankel{T}(convert(AbstractVector{T}, A.v), A.s)

# Size
size(H::Hankel)=H.s

# Retrieve an entry by two indices
function getindex(A::Hankel, i::Integer, j::Integer)
    @boundscheck checkbounds(A,i,j)
    return A.v[i+j-1]
end
similar(A::Hankel, T::Type, dims::Dims{2}) = Hankel{T}(similar(A.v, T, dims[1]+dims[2]-true), dims)
for fun in (:zero, :conj, :copy, :-, :similar, :real, :imag)
    @eval begin
        $fun(A::Hankel)=Hankel($fun(A.v), A.s)
    end
end
for op in (:+, :-)
    @eval begin
        function $op(A::Hankel,B::Hankel)
            promote_shape(A,B)
            Hankel($op(A.v,B.v),A.s)
        end
    end
end
function copyto!(A::Hankel, B::Hankel)
    promote_shape(A,B)
    copyto!(A.v,B.v)
end

transpose(A::Hankel) = Hankel(A.v,(A.s[2],A.s[1]))
adjoint(A::Hankel) = transpose(conj(A))
(==)(A::Hankel,B::Hankel) = A.v==B.v && A.s==B.s
function fill!(A::Hankel, x::Number)
    fill!(A.v,x)
    A
end
(*)(scalar::Number, C::Hankel) = Hankel(scalar * C.v, C.s)
(*)(C::Hankel,scalar::Number) = Hankel(C.v * scalar, C.s)

isconcrete(A::Hankel) = isconcretetype(A.v)

function reverse(A::Hankel; dims=1)
    if dims==1
        Toeplitz(reverse(A.vc), A.vr)
    elseif dims==2
        Toeplitz(A.v[A.s[2]:end], A.v[A.s[2]:-1:1])
    else
        throw(ArgumentError("invalid dimension $dims in reverse"))
    end
end
function reverse(A::AbstractToeplitz; dims=1)
    if dims==1
        Hankel(reverse(A.vc),A.vr)
    elseif dims==2
        Hankel(vcat(reverse(A.vr),A.vc), size(A))
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