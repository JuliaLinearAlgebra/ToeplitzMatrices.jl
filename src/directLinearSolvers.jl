# implementation of direct linear solves and inversion (for symmetric p.d. matrices at this point)
"""
    durbin(r::AbstractVector)

Computes `y = T \\ (-r)` where `T = SymmetricToeplitz(vcat(1, r[1:end-1]))`.
`T` is assumed to be positive definite. `r` is of length n.
"""
durbin(r::AbstractVector) = durbin!(zero(r), r)

"""
    durbin!(y::AbstractVector, r::AbstractVector)

Same as `durbin` but uses pre-allocated vector `y` to store the result.
"""
function durbin!(y::AbstractVector, r::AbstractVector)
    n = length(r)
    length(y) == n || throw(DimensionMismatch("length(y) = $(length(y)) ≠ $n = length(r)"))
    y[1] = -r[1]
    α, β = -r[1], one(eltype(r))
    for k in 1:n-1
        β *= (1-α^2)
        r_k, y_k = @views r[1:k], y[1:k]
        α = -(r[k+1] + reverse_dot(r_k, y_k)) / β
        reverse_increment!(y_k, y_k, α)
        y[k+1] = α
    end
    return y
end

"""
    trench(T::SymmetricToeplitz)

Trench's algorithm (see page 213 of Golub & van Loan) computes inverse of a
symmetric positive definite Toeplitz matrix in `O(n^2)` operations.
"""
function trench(T::SymmetricToeplitz)
    n = length(T.vc)
    B = zeros(eltype(T), n, n)
    trench!(B, T)
end

"""
    trench!(B::AbstractMatrix, T::SymmetricToeplitz)

Same as `trench` but uses `B` to store the upper triangular portion of the inverse
of `T` and returns a `Symmetric` wrapper of `B`.
"""
function trench!(B::AbstractMatrix, T::SymmetricToeplitz)
    r_0 = T.vc[1]
    r = @view T.vc[2:end]
    if !isone(r_0) # levinson implementation assumes normalization of diagonal
        r /= r_0 # not in place so that we don't mutate T
    end
    S = trench!(B, r)
    if !isone(r_0)
        @. B /= r_0
    end
    return S
end

# computes inverse of K = SymmetricToeplitz(vcat(1, r))
function trench(r::AbstractVector)
    n = length(r) + 1
    B = zeros(eltype(r), n, n)
    trench!(B, r)
end

# computes inverse of K = SymmetricToeplitz(vcat(1, r))
# uses B to store the inverse
# NOTE: only populates entries on upper triangle since the inverse is symmetric
function trench!(B::AbstractMatrix, r::AbstractVector)
    n = checksquare(B)
    n == length(r) + 1 || throw(DimensionMismatch())
    y = durbin(r)
    γ = inv(1 + dot(r, y))
    ν = γ * y[end:-1:1]
    B[1, 1] = γ
    @. B[1, 2:n] = γ * y
    for j in 2:n
        for i in 2:j
            @inbounds B[i, j] = B[i-1, j-1] + (ν[n+1-j] * ν[n+1-i] - ν[i-1] * ν[j-1]) / γ
        end
    end
    return Symmetric(B)
end

"""
    levinson(r::AbstractVector, b::AbstractVector)

Solves `x = T \\ b` where `T = SymmetricToeplitz(vcat(1, r))`, i.e. `T_{ij} = r[abs(i-j)]`
where by assumption `r[0] = 1` and `T` is positive definite.
"""
levinson(r::AbstractVector, b::AbstractVector) = levinson!(zero(b), r, b)

"""
    levinson!(x::AbstractVector, r::AbstractVector, b::AbstractVector, y::AbstractVector = zero(x))

Same as `levinson`, but uses the pre-allocated vector `x` to store the result.
"""
function levinson!(x::AbstractVector, r::AbstractVector, b::AbstractVector,
                   y::AbstractVector = zero(x))
    n = length(r) + 1
    length(x) == n || throw(DimensionMismatch("length(x) = $(length(x)) ≠ $n = length(r) + 1"))
    length(b) == n || throw(DimensionMismatch("length(b) = $(length(b)) ≠ $n = length(r) + 1"))
    y[1] = -r[1]
    x[1] = b[1]
    α, β = -r[1], one(eltype(r))
    @inbounds for k in 1:n-1
        β *= (1-α^2)
        r_k, x_k, y_k  = @views r[1:k], x[1:k], y[1:k]
        μ = (b[k+1] - reverse_dot(r_k, x_k)) / β
        reverse_increment!(x_k, y_k, μ)
        x[k+1] = μ
        if k < n-1
            α = -(r[k+1] + reverse_dot(r_k, y_k)) / β
            reverse_increment!(y_k, y_k, α) # computes y_k += α * reverse(y_k)
            y[k+1] = α
        end
    end
    return x
end

function levinson(T::SymmetricToeplitz, b::AbstractVector)
    r_0 = T.vc[1]
    r = @view T.vc[2:end]
    if !isone(r_0) # levinson implementation assumes normalization of diagonal
        r /= r_0
    end
    x = levinson(r, b)
    if !isone(r_0)
        @. x /= r_0
    end
    return x
end

# computes dot(x, reverse(y)) efficiently
function reverse_dot(x::AbstractArray, y::AbstractArray)
    n = length(x)
    n == length(y) || throw(DimensionMismatch())
    d = zero(promote_type(eltype(x), eltype(y)))
    @inbounds @simd for i in 1:n
        d += x[i] * y[n-i+1]
    end
    return d
end
# computes x += α * reverse(y) efficiently without temporary
# NOTE: works without allocations even when x === y
function reverse_increment!(x::AbstractArray, y::AbstractArray = x, α::Number = 1)
    n = length(x)
    n == length(y) || throw(DimensionMismatch())
    if x === y
        @inbounds @simd for i in 1:n÷2
            y_i = y[i] # important to store these for the iteration so that they aren't mutated
            z_i = y[n-i+1]
            x[i] += α * z_i
            x[n-i+1] += α * y_i
        end
        if isodd(n)
            i = (n÷2) + 1 # midpoint
            x[i] += α * y[i]
        end
    else
        @inbounds @simd for i in 1:n
            x[i] += α * y[n-i+1]
        end
    end
    return x
end
