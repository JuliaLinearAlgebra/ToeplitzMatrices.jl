# Tridiagonal Toeplitz eigen following Trench (1985)
# "On the eigenvalue problem for Toeplitz band matrices"
# https://www.sciencedirect.com/science/article/pii/0024379585902770

# Technically, these should be in FillArrays, but these were deemed to be too
# complex for that package, so they were shifted here
# See https://github.com/JuliaArrays/FillArrays.jl/pull/256

for MT in (:(Tridiagonal{<:Union{Real, Complex}, <:AbstractFillVector}),
            :(SymTridiagonal{<:Union{Real, Complex}, <:AbstractFillVector}),
            :(HermOrSym{T, <:Tridiagonal{T, <:AbstractFillVector{T}}} where {T<:Union{Real, Complex}})
            )
    @eval eigvals(A::$MT) = _eigvals_toeplitz(A)
end

___eigvals_toeplitz(a, sqrtbc, n) = [a + 2 * sqrtbc * cospi(q/(n+1)) for q in n:-1:1]

__eigvals_toeplitz(::AbstractMatrix, a, b, c, n) =
    ___eigvals_toeplitz(a, √(b*c), n)
__eigvals_toeplitz(::Union{SymTridiagonal, Symmetric{<:Any, <:Tridiagonal}}, a, b, c, n) =
    ___eigvals_toeplitz(a, b, n)
__eigvals_toeplitz(::Hermitian{<:Any, <:Tridiagonal}, a, b, c, n) =
    ___eigvals_toeplitz(real(a), abs(b), n)

# triangular Toeplitz
function _eigvals_toeplitz(T)
    require_one_based_indexing(T)
    n = checksquare(T)
    # extra care to handle 0x0 and 1x1 matrices
    # diagonal
    a = get(T, (1,1), zero(eltype(T)))
    # subdiagonal
    b = get(T, (2,1), zero(eltype(T)))
    # superdiagonal
    c = get(T, (1,2), zero(eltype(T)))
    vals = __eigvals_toeplitz(T, a, b, c, n)
    return vals
end

_eigvec_prefactor(A, cm1, c1, m) = sqrt(complex(cm1/c1))^m
_eigvec_prefactor(A::Union{SymTridiagonal, Symmetric{<:Any, <:Tridiagonal}}, cm1, c1, m) = oneunit(_eigvec_eltype(A))

function _eigvec_prefactors(A, cm1, c1)
    x = _eigvec_prefactor(A, cm1, c1, 1)
    [x^(j-1) for j in axes(A,1)]
end
_eigvec_prefactors(A::Union{SymTridiagonal, Symmetric{<:Any, <:Tridiagonal}}, cm1, c1) =
    Fill(_eigvec_prefactor(A, cm1, c1, 1), size(A,1))

_eigvec_eltype(A::SymTridiagonal) = float(eltype(A))
_eigvec_eltype(A) = complex(float(eltype(A)))

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
        qrev = n+1-q # match the default eigenvalue sorting
        for j in 1:cld(n,2)
            M[j, q] = prefactors[j] * sinpi(j*qrev/(n+1))
        end
        phase = iseven(n+q) ? 1 : -1
        for j in cld(n,2)+1:n
            M[j, q] = phase * prefactors[2j-n] * M[n+1-j,q]
        end
    end
    _normalizecols!(M, T)
    return M
end

for MT in (:(Tridiagonal{<:Union{Real,Complex}, <:AbstractFillVector}),
            :(SymTridiagonal{<:Union{Real,Complex}, <:AbstractFillVector}),
            :(HermOrSym{T, <:Tridiagonal{T, <:AbstractFillVector{T}}} where {T<:Union{Real,Complex}}),
            )

    @eval begin
        function eigvecs(A::$MT)
            _eigvecs_toeplitz(A)
        end
        function eigen(T::$MT)
            Eigen(eigvals(T), eigvecs(T))
        end
    end
end
