module IterativeLinearSolvers
    using LinearAlgebra
    import LinearAlgebra: Factorization
# Included from https://github.com/andreasnoack/IterativeLinearSolvers.jl
# Eventually, use IterativeSolvers.jl

Preconditioner{T} = Union{AbstractMatrix{T}, Factorization{T}}

function cg(A::AbstractMatrix{T},
    x::AbstractVector{T},
    b::AbstractVector{T},
    M::Preconditioner{T},
    max_it::Integer,
    tol::Real) where T<:LinearAlgebra.BlasReal
#  -- Iterative template routine --
#     Univ. of Tennessee and Oak Ridge National Laboratory
#     October 1, 1993
#     Details of this algorithm are described in "Templates for the
#     Solution of Linear Systems: Building Blocks for Iterative
#     Methods", Barrett, Berry, Chan, Demmel, Donato, Dongarra,
#     Eijkhout, Pozo, Romine, and van der Vorst, SIAM Publications,
#     1993. (ftp netlib2.cs.utk.edu; cd linalg; get templates.ps).
#
#  [x, error, iter, flag] = cg(A, x, b, M, max_it, tol)
#
# cg.m solves the symmetric positive definite linear system Ax=b
# using the Conjugate Gradient method with preconditioning.
#
# input   A        REAL symmetric positive definite matrix
#         x        REAL initial guess vector
#         b        REAL right hand side vector
#         M        REAL preconditioner matrix
#         max_it   INTEGER maximum number of iterations
#         tol      REAL error tolerance
#
# output  x        REAL solution vector
#         error    REAL error norm
#         iter     INTEGER number of iterations performed
#         flag     INTEGER: 0 = solution found to tolerance
#                           1 = no convergence given max_it

    n = length(b)
    flag = 0                                 # initialization
    iter = 0

    bnrm2 = norm(b)
    if bnrm2 == 0.0 bnrm2 = one(T) end

    local ρ₁
    z = zeros(T, n)
    q = zeros(T, n)
    p = zeros(T, n)
    # r = copy(b)
    # mul!(r,A,x,-one(T),one(T))
    r = b - A*x
    error = norm(r)/bnrm2
    if error < tol
        return
    end

    for iter_inner = 1:max_it                       # begin iteration
        iter = iter_inner

        z[:] = r
        ldiv!(M, z)
        # z[:] = M\r
        ρ = dot(r,z)

        if iter > 1                       # direction vector
            β = ρ/ρ₁
            for l = 1:n
                p[l] = z[l] + β*p[l]
            end
        else
            p[:] = z
        end

        # mul!(q,A,p,one(T),zero(T))
        q[:] = A*p
        α = ρ / dot(p,q)
        for l = 1:n
            x[l] += α*p[l]                    # update approximation vector
            r[l] -= α*q[l]                    # compute residual
        end

        error = norm(r)/bnrm2                     # check convergence
        if error <= tol
            break
        end

        ρ₁ = ρ

    end

    if error > tol flag = 1 end                 # no convergence
    return x, error, iter, flag
end

function cgs(A::AbstractMatrix{T},
    x::AbstractVector{T},
    b::AbstractVector{T},
    M::Preconditioner{T},
    max_it::Integer,
    tol::Real) where T<:Number
#  -- Iterative template routine --
#     Univ. of Tennessee and Oak Ridge National Laboratory
#     October 1, 1993
#     Details of this algorithm are described in "Templates for the
#     Solution of Linear Systems: Building Blocks for Iterative
#     Methods", Barrett, Berry, Chan, Demmel, Donato, Dongarra,
#     Eijkhout, Pozo, Romine, and van der Vorst, SIAM Publications,
#     1993. (ftp netlib2.cs.utk.edu; cd linalg; get templates.ps).
#
#  [x, error, iter, flag] = cgs(A, x, b, M, max_it, tol)
#
# cgs.m solves the linear system Ax=b using the
# Conjugate Gradient Squared Method with preconditioning.
#
# input   A        REAL matrix
#         x        REAL initial guess vector
#         b        REAL right hand side vector
#         M        REAL preconditioner
#         max_it   INTEGER maximum number of iterations
#         tol      REAL error tolerance
#
# output  x        REAL solution vector
#         error    REAL error norm
#         iter     INTEGER number of iterations performed
#         flag     INTEGER: 0 = solution found to tolerance
#                           1 = no convergence given max_it

    iter = 0                               # initialization
    flag = 0

    n = length(b)
    bnrm2 = norm(b)
    if bnrm2 == 0
        bnrm2 = one(bnrm2)
    end

    u = zeros(T, n)
    p = zeros(T, n)
    p̂ = zeros(T, n)
    q = zeros(T, n)
    û = zeros(T, n)
    v̂ = zeros(T, n)
    ρ = zero(T)
    ρ₁ = ρ
    r = copy(b)
    mul!(r, A, x, -one(T), one(T))
    # r = b - A*x
    error = norm(r)/bnrm2

    if error < tol
        return x, error, iter, flag
    end

    r_tld = copy(r)

    for iter_inner = 1:max_it                      # begin iteration
        iter = iter_inner

        ρ = dot(r_tld, r)
        if ρ == 0
            break
        end

        if iter > 1                          # direction vectors
            β = ρ/ρ₁
            for l = 1:n
                u[l] = r[l] + β*q[l]
                p[l] = u[l] + β*(q[l] + β*p[l])
            end
        else
            u[:] = r
            p[:] = u
        end

        p̂[:] = p
        ldiv!(M, p̂)
        # p̂[:] = M\p
        mul!(v̂, A, p̂, one(T), zero(T))  # adjusting scalars
        # v̂[:] = A*p̂
        α = ρ/dot(r_tld, v̂)
        for l = 1:n
            q[l] = u[l] - α*v̂[l]
            û[l] = u[l] + q[l]
        end
        ldiv!(M, û)
        # û[:] = M\û

        for l = 1:n
            x[l] += α*û[l]                  # update approximation
        end

        mul!(r, A, û, -α, one(T))
        # r[:] -= α*(A*û)
        error = norm(r)/bnrm2               # check convergence
        if error <= tol
            break
        end

        ρ₁ = ρ

    end

    if error <= tol                         # converged
        flag = 0
    elseif ρ == 0                           # breakdown
        flag = -1
    else                                    # no convergence
        flag = 1
    end
    return x, error, iter, flag
end

end
