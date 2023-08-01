# Tridiagonal Toeplitz eigen following Trench (1985)
# "On the eigenvalue problem for Toeplitz band matrices"
# https://www.sciencedirect.com/science/article/pii/0024379585902770

# Technically, these should be in FillArrays, but these were deemed to be too
# complex for that package, so they were shifted here
# See https://github.com/JuliaArrays/FillArrays.jl/pull/256

# The methods aren't defined for general Tridiagonal or Hermitian, as the
# ordering of eigenvectors needs fixing
for MT in (:(SymTridiagonal{<:Union{Real,Complex}, <:AbstractFillVector}),
            :(Symmetric{T, <:Tridiagonal{T, <:AbstractFillVector{T}}} where {T<:Union{Real,Complex}})
            )
    @eval function eigvals(A::$MT)
        n = size(A,1)
        if n <= 2 # repeated roots possible
            eigvals(Matrix(A))
        else
            _eigvals(A)
        end
    end
end

for MT in (:(SymTridiagonal{<:Union{Real,Complex}, <:AbstractFillVector}),
            :(Symmetric{T, <:Tridiagonal{T, <:AbstractFillVector{T}}} where {T<:Union{Real,Complex}}),
            )

    @eval begin
        eigvecs(A::$MT) = _eigvecs(A)
        eigen(A::$MT) = _eigen(A)
    end
end


___eigvals(a, sqrtbc, n) = [a + 2 * sqrtbc * cospi(q/(n+1)) for q in n:-1:1]

__eigvals(::AbstractMatrix, a, b, c, n) =
    ___eigvals(a, √(complex(b*c)), n)
__eigvals(::Union{SymTridiagonal, Symmetric{<:Any, <:Tridiagonal}}, a, b, c, n) =
    ___eigvals(a, b, n)
__eigvals(::Hermitian{<:Any, <:Tridiagonal}, a, b, c, n) =
    ___eigvals(real(a), abs(b), n)

# triangular Toeplitz
function _eigvals(T)
    require_one_based_indexing(T)
    n = checksquare(T)
    # extra care to handle 0x0 and 1x1 matrices
    # diagonal
    a = get(T, (1,1), zero(eltype(T)))
    # subdiagonal
    b = get(T, (2,1), zero(eltype(T)))
    # superdiagonal
    c = get(T, (1,2), zero(eltype(T)))
    vals = __eigvals(T, a, b, c, n)
    return vals
end

_eigvec_eltype(A::Union{SymTridiagonal,
                    Symmetric{<:Any,<:Tridiagonal}}) = float(eltype(A))
_eigvec_eltype(A) = complex(float(eltype(A)))

_eigvec_prefactor(A, cm1, c1, m) = sqrt(complex(cm1/c1))^m
_eigvec_prefactor(A::Union{SymTridiagonal, Symmetric{<:Any, <:Tridiagonal}}, cm1, c1, m) = oneunit(_eigvec_eltype(A))

function _eigvec_prefactors(A, cm1, c1)
    x = _eigvec_prefactor(A, cm1, c1, 1)
    [x^(j-1) for j in axes(A,1)]
end
_eigvec_prefactors(A::Union{SymTridiagonal, Symmetric{<:Any, <:Tridiagonal}}, cm1, c1) =
    Fill(_eigvec_prefactor(A, cm1, c1, 1), size(A,1))

@static if !isdefined(Base, :eachcol)
    eachcol(A) = (view(A,:,i) for i in axes(A,2))
end

_normalizecols!(M, T) = foreach(normalize!, eachcol(M))
function _normalizecols!(M, T::Union{SymTridiagonal, Symmetric{<:Number, <:Tridiagonal}})
    n = size(M,1)
    invnrm = sqrt(2/(n+1))
    M .*= invnrm
    return M
end

function _eigvecs_toeplitz(T)
    require_one_based_indexing(T)
    n = checksquare(T)
    M = Matrix{_eigvec_eltype(T)}(undef, n, n)
    n == 0 && return M
    n == 1 && return fill!(M, oneunit(eltype(M)))
    cm1 = T[2,1] # subdiagonal
    c1 = T[1,2]  # superdiagonal
    prefactors = _eigvec_prefactors(T, cm1, c1)
    for q in axes(M,2)
        for j in 1:cld(n,2)
            jphase = 2isodd(j) - 1
            M[j, q] = prefactors[j] * jphase * sinpi(j * q/(n+1))
        end
        phase = iseven(n+q) ? 1 : -1
        for j in cld(n,2)+1:n
            M[j, q] = phase * prefactors[2j-n] * M[n+1-j,q]
        end
    end
    _normalizecols!(M, T)
    return M
end

function _eigvecs_toeplitz(T::Union{SymTridiagonal, Symmetric{<:Any,<:Tridiagonal}})
    require_one_based_indexing(T)
    n = checksquare(T)
    M = Matrix{_eigvec_eltype(T)}(undef, n, n)
    n == 0 && return M
    n == 1 && return fill!(M, oneunit(eltype(M)))
    for q in 1:cld(n,2)
        for j in 1:q
            jphase = 2isodd(j) - 1
            M[j, q] = jphase * sinpi(j * q/(n+1))
        end
    end
    for q in 1:cld(n,2)
        for j in q+1:cld(n,2)
            qphase = 2isodd(q) - 1
            jphase = 2isodd(j) - 1
            phase = qphase * jphase
            M[j, q] = phase * M[q, j]
        end
    end
    for q in 1:cld(n,2)
        phase = iseven(n+q) ? 1 : -1
        for j in cld(n,2)+1:n
            M[j, q] = phase * M[n+1-j,q]
        end
    end
    for q in cld(n,2)+1:n
        for j in 1:cld(n,2)
            qphase = 2isodd(q) - 1
            jphase = 2isodd(j) - 1
            phase = qphase * jphase
            M[j, q] = phase * M[q, j]
        end
        phase = iseven(n+q) ? 1 : -1
        for j in cld(n,2)+1:n
            M[j, q] = phase * M[n+1-j,q]
        end
    end
    _normalizecols!(M, T)
    return M
end

function _eigvecs(A)
    n = size(A,1)
    if n <= 2 # repeated roots possible
        eigvecs(Matrix(A))
    else
        _eigvecs_toeplitz(A)
    end
end

function _eigen(A)
    n = size(A,1)
    if n <= 2 # repeated roots possible
        eigen(Matrix(A))
    else
        Eigen(eigvals(A), eigvecs(A))
    end
end
