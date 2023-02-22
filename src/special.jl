# special Toeplitz types that can be represented by a single vector
# Symmetric, Circulant, LowerTriangular, UpperTriangular
for TYPE in (:SymmetricToeplitz, :Circulant, :LowerTriangularToeplitz, :UpperTriangularToeplitz)
    @eval begin
        struct $TYPE{T<:Number} <: AbstractToeplitz{T}
            v::AbstractVector{T}
        end

        convert(::Type{AbstractToeplitz{T}}, A::$TYPE) where {T} = convert($TYPE{T},A)
        convert(::Type{$TYPE{T}}, A::$TYPE) where {T} = $TYPE(convert(AbstractVector{T},A.v))

        function size(A::$TYPE, dim::Int)
            if dim < 1
                error("arraysize: dimension out of range")
            end
            return dim < 3 ? length(A.v) : 1
        end
        
        adjoint(A::$TYPE) = transpose(conj(A))
        (*)(scalar::Number, C::$TYPE) = $TYPE(scalar * C.v)
        (*)(C::$TYPE,scalar::Number) = $TYPE(C.v * scalar)
        (==)(A::$TYPE,B::$TYPE) = A.v==B.v

        function fill!(A::$TYPE, x::Number)
            fill!(A.v,x)
            A
        end
    end
    for fun in (:zero, :conj, :copy, :-, :similar)
        @eval begin
            $fun(A::$TYPE)=$TYPE($fun(A.v))
        end
    end
    for op in (:+, :-, :copyto!)
        @eval begin
            $op(A::$TYPE,B::$TYPE)=$TYPE($op(A.v,B.v))
        end
    end
    for TY in (:AbstractVector, :AbstractMatrix, :AbstractToeplitz)
        @eval begin
            $TYPE(v::$TY) = $TYPE{eltype(v)}(v)
        end
    end
end
TriangularToeplitz{T}=Union{UpperTriangularToeplitz{T},LowerTriangularToeplitz{T}}

# vc and vr
function getproperty(A::SymmetricToeplitz, s::Symbol)
    if s==:vc || s==:vr
        getfield(A,:v)
    else
        getfield(A,s)
    end
end
function getproperty(A::Circulant, s::Symbol)
    if s==:vc
        getfield(A,:v)
    elseif s==:vr
        _circulate(getfield(A,:v))
    else
        getfield(A,s)
    end
end
function getproperty(A::LowerTriangularToeplitz, s::Symbol)
    if s==:vc
        getfield(A,:v)
    elseif s==:vr
        _firstnonzero(getfield(A,:v))
    elseif s==:uplo
        :L
    else
        getfield(A,s)
    end
end
function getproperty(A::UpperTriangularToeplitz, s::Symbol)
    if s==:vr
        getfield(A,:v)
    elseif s==:vc
        _firstnonzero(getfield(A,:v))
    elseif s==:uplo
        :U
    else
        getfield(A,s)
    end
end

_circulate(v::AbstractVector) = vcat(v[1],v[end:-1:2])
_firstnonzero(v::AbstractVector) = vcat(v[1],zero(view(v,2:lastindex(v))))

# transpose
transpose(A::SymmetricToeplitz) = A
transpose(A::Circulant) = Circulant(A.vr)
transpose(A::LowerTriangularToeplitz) = UpperTriangularToeplitz(A.v)
transpose(A::UpperTriangularToeplitz) = LowerTriangularToeplitz(A.v)

# getindex
function getindex(A::SymmetricToeplitz, i::Integer, j::Integer)
    @boundscheck checkbounds(A,i,j)
    return A.v[abs(i - j) + 1]
end
function getindex(C::Circulant, i::Integer, j::Integer)
    @boundscheck checkbounds(C,i,j)
    d = i - j
    return C.v[d < 0 ? size(C,1)+d+1 : d+1]
end
function getindex(A::LowerTriangularToeplitz{T}, i::Integer, j::Integer) where T
    @boundscheck checkbounds(A,i,j)
    return i >= j ? A.v[i - j + 1] : zero(T)
end
function getindex(A::UpperTriangularToeplitz{T}, i::Integer, j::Integer) where T
    @boundscheck checkbounds(A,i,j)
    return i <= j ? A.v[j - i + 1] : zero(T)
end

# constructors
function Circulant{T}(v::AbstractVector, uplo::Symbol) where T<:Number
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
function Circulant{T}(A::AbstractMatrix, uplo::Symbol = :L) where T<:Number
    if uplo == :L
        Circulant{T}(_vc(A), uplo)
    elseif uplo == :U
        Circulant{T}(_vr(A), uplo)
    else
        throw(ArgumentError("expected :L or :U; get $uplo."))
    end
end
Circulant(A::AbstractMatrix, uplo::Symbol) = Circulant{eltype(A)}(A,uplo)
function SymmetricToeplitz{T}(A::AbstractMatrix, uplo::Symbol = :U) where T<:Number
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

Circulant{T}(A::AbstractMatrix) where T<:Number = Circulant{T}(A, :L)
UpperTriangularToeplitz{T}(A::AbstractMatrix) where T<:Number = UpperTriangularToeplitz{T}(_vr(A))
LowerTriangularToeplitz{T}(A::AbstractMatrix) where T<:Number = LowerTriangularToeplitz{T}(_vc(A))
_toeplitztype(s::Symbol) = Symbol(s,"Toeplitz")
for TYPE in (:UpperTriangular, :LowerTriangular)
    @eval begin
        $TYPE{T}(A::AbstractToeplitz) where T<:Number = $(_toeplitztype(TYPE)){T}(A)
        $TYPE(A::AbstractToeplitz) = $TYPE{eltype(A)}(A)
        convert(::Type{TriangularToeplitz{T}},A::$(_toeplitztype(TYPE))) where T<:Number = convert($(_toeplitztype(TYPE)){T},A)
    end
end

# Triangular
for TYPE in (:AbstractMatrix, :AbstractVector)
    @eval begin
        TriangularToeplitz(A::$TYPE, uplo::Symbol) = TriangularToeplitz{eltype(A)}(A, uplo)
        function TriangularToeplitz{T}(A::$TYPE, uplo::Symbol) where T<:Number
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


# circulant
const CirculantFactorization{T<:Number} = ToeplitzFactorization{T,Circulant{T}}
function factorize(C::Circulant)
    T = eltype(C)
    vc = C.vc
    S = promote_type(T, Complex{Float32})
    tmp = Vector{S}(undef, length(vc))
    copyto!(tmp, vc)
    dft = plan_fft!(tmp)
    return ToeplitzFactorization{T,typeof(C),S,typeof(dft)}(dft * tmp, similar(tmp), dft)
end

Base.:*(A::Circulant, B::Circulant) = factorize(A) * factorize(B)
Base.:*(A::CirculantFactorization, B::Circulant) = A * factorize(B)
Base.:*(A::Circulant, B::CirculantFactorization) = factorize(A) * B
function Base.:*(A::CirculantFactorization, B::CirculantFactorization)
    A_vcvr_dft = A.vcvr_dft
    B_vcvr_dft = B.vcvr_dft
    m = length(A_vcvr_dft)
    n = length(B_vcvr_dft)
    if m != n
        throw(DimensionMismatch(
            "size of matrix A, $(m)x$(m), does not match size of matrix B, $(n)x$(n)"
        ))
    end

    vc = A.dft \ (A_vcvr_dft .* B_vcvr_dft)

    return Circulant(maybereal(Base.promote_eltype(A, B), vc))
end

# Make an eager adjoint, similar to adjoints of Diagonal in LinearAlgebra
adjoint(C::CirculantFactorization{T,S,P}) where {T,S,P} =
    CirculantFactorization{T,S,P}(conj.(C.vcvr_dft), C.tmp, C.dft)
Base.:*(A::Adjoint{<:Any,<:Circulant}, B::Circulant) = factorize(parent(A))' * factorize(B)
Base.:*(A::Adjoint{<:Any,<:Circulant}, B::CirculantFactorization) =
    factorize(parent(A))' * B
Base.:*(A::Adjoint{<:Any,<:Circulant}, B::Adjoint{<:Any,<:Circulant}) =
    factorize(parent(A))' * factorize(parent(B))'
Base.:*(A::Circulant, B::Adjoint{<:Any,<:Circulant}) = factorize(A) * factorize(parent(B))'
Base.:*(A::CirculantFactorization, B::Adjoint{<:Any,<:Circulant}) = A * factorize(parent(B))'

# Triangular
function _tridiff!(A::TriangularToeplitz, k::Integer)
    if k >= 0
        if isconcretetype(typeof(A.v))
            for i in k+2:lastindex(A.v)
                A.v[i] = zero(eltype(A))
            end
        else
            A.v=vcat(A.v[1:k+1], zero(A.v[k+2:end]))
        end
    else
        fill!(A,zero(eltype(A)))
    end
    A
end
tril!(A::UpperTriangularToeplitz, k::Integer) = _tridiff!(A,k)
triu!(A::LowerTriangularToeplitz, k::Integer) = _tridiff!(A,-k)

function _trisame!(A::TriangularToeplitz, k::Integer)
    if k < 0
        if isconcretetype(typeof(A.v))
            for i in 1:-k
                A.v[i]=zero(eltype(A))
            end
        else
            A.v=vcat(A.v[1:-k+1], zero(A.v[-k+2:end]))
        end
    end
    A
end
tril!(A::LowerTriangularToeplitz, k::Integer) = _trisame!(A,k)
triu!(A::UpperTriangularToeplitz, k::Integer) = _trisame!(A,-k)

# Hankel
struct Hankel{T<:Number} <: AbstractMatrix{T}
    v::AbstractVector{T}
    s::NTuple{2,Integer} # size
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
        if s[1] == ℵ₀
            Hankel(A[:,1],s)
        else
            Hankel(A[:,1],A[end,:])
        end
    elseif uplo == :U
        if s[2] == ℵ₀
            Hankel(A[1,:],s)
        else
            Hankel(A[1,:],A[:,end])
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

for fun in (:zero, :conj, :copy, :-, :similar)
    @eval begin
        $fun(A::Hankel)=Hankel($fun(A.v), A.s)
    end
end
for op in (:+, :-, :copyto!)
    @eval begin
        function $op(A::Hankel,B::Hankel)
            promote_shape(A,B)
            Hankel($op(A.v,B.v),A.s)
        end
    end
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