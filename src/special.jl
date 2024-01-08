# special Toeplitz types that can be represented by a single vector
# Symmetric, Circulant, LowerTriangular, UpperTriangular

abstract type AbstractToeplitzSingleVector{T} <: AbstractToeplitz{T} end

parent(A::AbstractToeplitzSingleVector) = A.v
basetype(x) = basetype(typeof(x))

function size(A::AbstractToeplitzSingleVector)
    n = length(parent(A))
    (n,n)
end

adjoint(A::AbstractToeplitzSingleVector) = transpose(conj(A))
function zero!(A::AbstractToeplitzSingleVector, inds = eachindex(parent(A)))
    zero!(parent(A), inds)
    return A
end

function lmul!(x::Number, A::AbstractToeplitzSingleVector)
    lmul!(x,parent(A))
    A
end
function rmul!(A::AbstractToeplitzSingleVector, x::Number)
    rmul!(parent(A),x)
    A
end

for fun in (:iszero,)
    @eval $fun(A::AbstractToeplitzSingleVector) = $fun(parent(A))
end

AbstractToeplitz{T}(A::AbstractToeplitzSingleVector) where T = basetype(A){T}(A)

(*)(scalar::Number, C::AbstractToeplitzSingleVector) = basetype(C)(scalar * parent(C))
(*)(C::AbstractToeplitzSingleVector, scalar::Number) = basetype(C)(parent(C) * scalar)

AbstractMatrix{T}(A::AbstractToeplitzSingleVector) where {T} = basetype(A){T}(AbstractVector{T}(A.v))

for fun in (:zero, :conj, :copy, :-, :real, :imag)
    @eval $fun(A::AbstractToeplitzSingleVector) = basetype(A)($fun(parent(A)))
end

for TYPE in (:SymmetricToeplitz, :Circulant, :LowerTriangularToeplitz, :UpperTriangularToeplitz)
    @eval begin
        struct $TYPE{T, V<:AbstractVector{T}} <: AbstractToeplitzSingleVector{T}
            v::V
            function $TYPE{T,V}(v::V) where {T,V<:AbstractVector{T}}
                require_one_based_indexing(v)
                new{T,V}(v)
            end
        end
        function $TYPE{T}(v::AbstractVector) where T
            vT = convert(AbstractVector{T},v)
            $TYPE{T, typeof(vT)}(vT)
        end
        $TYPE(v::V) where {T,V<:AbstractVector{T}} = $TYPE{T,V}(v)

        basetype(::Type{T}) where {T<:$TYPE} = $TYPE

        (==)(A::$TYPE, B::$TYPE) = A.v == B.v

        convert(::Type{$TYPE{T}}, A::$TYPE) where {T} = A isa $TYPE{T} ? A : $TYPE{T}(A)::$TYPE{T}

        function copyto!(A::$TYPE,B::$TYPE)
            copyto!(A.v,B.v)
            A
        end
    end
    for op in (:+, :-)
        @eval $op(A::$TYPE,B::$TYPE) = $TYPE($op(A.v,B.v))
    end
    for TY in (:AbstractMatrix, :AbstractToeplitz)
        @eval $TYPE(v::$TY) = $TYPE{eltype(v)}(v)
    end
end
TriangularToeplitz{T}=Union{UpperTriangularToeplitz{T},LowerTriangularToeplitz{T}}

# vc and vr
function getproperty(A::SymmetricToeplitz, s::Symbol)
    if s == :vc || s == :vr
        getfield(A,:v)
    else
        getfield(A,s)
    end
end
function getproperty(A::Circulant, s::Symbol)
    if s == :vc
        getfield(A,:v)
    elseif s == :vr
        _circulate(getfield(A,:v))
    else
        getfield(A,s)
    end
end
function getproperty(A::LowerTriangularToeplitz, s::Symbol)
    if s == :vc
        getfield(A,:v)
    elseif s == :vr
        _firstnonzero(getfield(A,:v))
    elseif s == :uplo
        :L
    else
        getfield(A,s)
    end
end
function getproperty(A::UpperTriangularToeplitz, s::Symbol)
    if s == :vr
        getfield(A,:v)
    elseif s == :vc
        _firstnonzero(getfield(A,:v))
    elseif s == :uplo
        :U
    else
        getfield(A,s)
    end
end

_circulate(v::AbstractVector) = view(Circulant(v), 1, :)
function _firstnonzero(v::AbstractVector)
    w = zero(v)
    w[1] = v[1]
    w
end

# transpose
transpose(A::SymmetricToeplitz) = A
transpose(A::Circulant) = Circulant(A.vr)
transpose(A::LowerTriangularToeplitz) = UpperTriangularToeplitz(A.v)
transpose(A::UpperTriangularToeplitz) = LowerTriangularToeplitz(A.v)

# getindex
Base.@propagate_inbounds function getindex(A::SymmetricToeplitz, i::Integer, j::Integer)
    @boundscheck checkbounds(A,i,j)
    return A.v[abs(i - j) + 1]
end
Base.@propagate_inbounds function getindex(C::Circulant, i::Integer, j::Integer)
    @boundscheck checkbounds(C,i,j)
    d = i - j
    return C.v[d < 0 ? size(C,1)+d+1 : d+1]
end
Base.@propagate_inbounds function getindex(A::LowerTriangularToeplitz{T}, i::Integer, j::Integer) where T
    @boundscheck checkbounds(A,i,j)
    return i >= j ? A.v[i - j + 1] : zero(T)
end
Base.@propagate_inbounds function getindex(A::UpperTriangularToeplitz{T}, i::Integer, j::Integer) where T
    @boundscheck checkbounds(A,i,j)
    return i <= j ? A.v[j - i + 1] : zero(T)
end

# constructors
function Circulant{T}(v::AbstractVector, uplo::Symbol) where T
    if uplo == :L
        Circulant{T}(v)
    elseif uplo == :U
        Circulant{T}(_circulate(v))
    else
        throw(ArgumentError("expected :L or :U; got $uplo."))
    end
end
Circulant(v::AbstractVector, uplo::Symbol) = Circulant{eltype(v)}(v,uplo)
# from AbstractMatrix, AbstractToeplitz
function Circulant{T}(A::AbstractMatrix, uplo::Symbol = :L) where T
    checksquare(A)
    if uplo == :L
        Circulant{T}(_vc(A), uplo)
    elseif uplo == :U
        Circulant{T}(_vr(A), uplo)
    else
        throw(ArgumentError("expected :L or :U; get $uplo."))
    end
end
Circulant(A::AbstractMatrix, uplo::Symbol) = Circulant{eltype(A)}(A,uplo)
function SymmetricToeplitz{T}(A::AbstractMatrix, uplo::Symbol = :U) where T
    checksquare(A)
    if uplo == :L
        SymmetricToeplitz{T}(_vc(A))
    elseif uplo == :U
        SymmetricToeplitz{T}(_vr(A))
    else
        throw(ArgumentError("expected :L or :U; got $uplo."))
    end
end
SymmetricToeplitz(A::AbstractMatrix, uplo::Symbol) = SymmetricToeplitz{eltype(A)}(A,uplo)
Symmetric(A::AbstractToeplitz, uplo::Symbol = :U) = SymmetricToeplitz(A,uplo)

function UpperTriangularToeplitz{T}(A::AbstractMatrix) where T
    checksquare(A)
    UpperTriangularToeplitz{T}(_vr(A))
end
function LowerTriangularToeplitz{T}(A::AbstractMatrix) where T
    checksquare(A)
    LowerTriangularToeplitz{T}(_vc(A))
end
_toeplitztype(s::Symbol) = Symbol(s,"Toeplitz")
for TYPE in (:UpperTriangular, :LowerTriangular)
    @eval begin
        $TYPE{T}(A::AbstractToeplitz) where T = $(_toeplitztype(TYPE)){T}(A)
        $TYPE(A::AbstractToeplitz) = $TYPE{eltype(A)}(A)
        convert(::Type{TriangularToeplitz{T}},A::$(_toeplitztype(TYPE))) where T<:Number = convert($(_toeplitztype(TYPE)){T},A)
    end
end

_copymutable(A::AbstractToeplitzSingleVector) = basetype(A)(_copymutable(parent(A)))

# Triangular
for TYPE in (:AbstractMatrix, :AbstractVector)
    @eval begin
        TriangularToeplitz(A::$TYPE, uplo::Symbol) = TriangularToeplitz{eltype(A)}(A, uplo)
        function TriangularToeplitz{T}(A::$TYPE, uplo::Symbol) where T
            if uplo == :L
                LowerTriangularToeplitz{T}(A)
            elseif uplo == :U
                UpperTriangularToeplitz{T}(A)
            else
                throw(ArgumentError("expected :L or :U. got $uplo."))
            end
        end
    end
end

# tril and triu
function _tridiff!(A::TriangularToeplitz, k::Integer)
    i1, iend = firstindex(A.v), lastindex(A.v)
    inds = max(i1, k+2):iend
    if k >= 0
        zero!(A, inds)
    else
        zero!(A)
    end
    A
end
tril!(A::UpperTriangularToeplitz, k::Integer=0) = _tridiff!(A,k)
triu!(A::LowerTriangularToeplitz, k::Integer=0) = _tridiff!(A,-k)

function _trisame!(A::TriangularToeplitz, k::Integer)
    i1, iend = firstindex(A.v), lastindex(A.v)
    inds = i1:min(-k,iend)
    if k < 0
        zero!(A, inds)
    end
    A
end
tril!(A::LowerTriangularToeplitz, k::Integer=0) = _trisame!(A,k)
triu!(A::UpperTriangularToeplitz, k::Integer=0) = _trisame!(A,-k)

tril(A::TriangularToeplitz, k::Integer=0) = tril!(_copymutable(A), k)
triu(A::TriangularToeplitz, k::Integer=0) = triu!(_copymutable(A), k)

isdiag(A::Union{Circulant, LowerTriangularToeplitz, SymmetricToeplitz}) = all(iszero, @view _vc(A)[2:end])
isdiag(A::UpperTriangularToeplitz) = all(iszero, @view _vr(A)[2:end])
