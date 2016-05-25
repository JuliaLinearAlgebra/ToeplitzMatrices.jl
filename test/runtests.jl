using Base.Test
using ToeplitzMatrices

ns = 101
nl = 2000

xs = randn(ns, 5)
xl = randn(nl, 5)

@printf("General Toeplitz: ")
# Square
As = Toeplitz(0.9.^(0:ns-1), 0.4.^(0:ns-1))
Al = Toeplitz(0.9.^(0:nl-1), 0.4.^(0:nl-1))
@test_approx_eq As * xs full(As) * xs
@test_approx_eq Al * xl full(Al) * xl
@test_approx_eq A_ldiv_B!(As, copy(xs)) full(As) \ xs
@test_approx_eq A_ldiv_B!(Al, copy(xl)) full(Al) \ xl

# Rectangular
Ar1 = Toeplitz(0.9.^(0:nl-1), 0.4.^(0:ns-1))
Ar2 = Toeplitz(0.9.^(0:ns-1), 0.4.^(0:nl-1))
@test_approx_eq Ar1 * xs full(Ar1) * xs
@test_approx_eq Ar2 * xl full(Ar2) * xl
@printf("OK!\n")

@printf("Symmetric Toeplitz: ")
As = SymmetricToeplitz(0.9.^(0:ns-1))
Ab = SymmetricToeplitz(abs(randn(ns)))
Al = SymmetricToeplitz(0.9.^(0:nl-1))
@test_approx_eq As * xs full(As) * xs
@test_approx_eq Ab * xs full(Ab) * xs
@test_approx_eq Al * xl full(Al) * xl
@test_approx_eq A_ldiv_B!(As, copy(xs)) full(As) \ xs
@test_approx_eq A_ldiv_B!(Ab, copy(xs)) full(Ab) \ xs
@test_approx_eq A_ldiv_B!(Al, copy(xl)) full(Al) \ xl
@test_approx_eq StatsBase.levinson(As, xs) full(As) \ xs
@test_approx_eq StatsBase.levinson(Ab, xs) full(Ab) \ xs
@test_approx_eq StatsBase.levinson(Al, xl) full(Al) \ xl
@printf("OK!\n")

@printf("Circulant: ")
As = Circulant(0.9.^(0:ns-1))
Al = Circulant(0.9.^(0:nl-1))
@test_approx_eq As * xs full(As) * xs
@test_approx_eq Al * xl full(Al) * xl
@test_approx_eq A_ldiv_B!(As, copy(xs)) full(As) \ xs
@test_approx_eq A_ldiv_B!(Al, copy(xl)) full(Al) \ xl
@printf("OK!\n")

@printf("Upper triangular Toeplitz: ")
As = TriangularToeplitz(0.9.^(0:ns - 1), :U)
Al = TriangularToeplitz(0.9.^(0:nl - 1), :U)
@test_approx_eq As * xs full(As) * xs
@test_approx_eq Al * xl full(Al) * xl
@test_approx_eq A_ldiv_B!(As, copy(xs)) full(As) \ xs
@test_approx_eq A_ldiv_B!(Al, copy(xl)) full(Al) \ xl
@printf("OK!\n")

@printf("Lower triangular Toeplitz: ")
As = TriangularToeplitz(0.9.^(0:ns - 1), :L)
Al = TriangularToeplitz(0.9.^(0:nl - 1), :L)
@test_approx_eq As * xs full(As) * xs
@test_approx_eq Al * xl full(Al) * xl
@test_approx_eq A_ldiv_B!(As, copy(xs)) full(As) \ xs
@test_approx_eq A_ldiv_B!(Al, copy(xl)) full(Al) \ xl
@printf("OK!\n")


@printf("Hankel: ")
H=Hankel([1.0,2,3,4,5],[5.0,6,7,8,0])
x=ones(5)
@test_approx_eq full(H)*x H*x
@printf("OK!\n")


if isdir(Pkg.dir("FastTransforms"))
    @printf("BigFloat: ")
    using FastTransforms
    T=Toeplitz(BigFloat[1,2,3,4,5],BigFloat[1,6,7,8,0])
    @test_approx_eq T*ones(BigFloat,5) [22,24,19,16,15]

    n=512
    r=map(BigFloat,rand(n))
    T=Toeplitz(r,[r[1];map(BigFloat,rand(n-1))])
    @test_approx_eq T*ones(BigFloat,n) full(T)*ones(BigFloat,n)

    T=TriangularToeplitz(BigFloat[1,2,3,4,5],:L)
    @test_approx_eq T*ones(BigFloat,5) full(T)*ones(BigFloat,5)

    n=512
    r=map(BigFloat,rand(n))
    T=TriangularToeplitz(r,:L)
    @test_approx_eq T*ones(BigFloat,n) full(T)*ones(BigFloat,n)

    T=TriangularToeplitz(BigFloat[1,2,3,4,5],:U)
    @test_approx_eq T*ones(BigFloat,5) full(T)*ones(BigFloat,5)

    n=512
    r=map(BigFloat,rand(n))
    T=TriangularToeplitz(r,:U)
    @test_approx_eq T*ones(BigFloat,n) full(T)*ones(BigFloat,n)
end
@printf("OK!\n")
