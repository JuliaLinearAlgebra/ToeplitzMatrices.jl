using Base.Test
using ToeplitzMatrices

xs = randn(10)
xl = randn(1000)

@printf("General Toeplitz: ")
As = Toeplitz(0.9.^[0:9], 0.4.^[0:9])
Al = Toeplitz(0.9.^[0:999], 0.4.^[0:999])
@test_approx_eq As*xs full(As)*xs
@test_approx_eq Al*xl full(Al)*xl
@test_approx_eq solve!(As,copy(xs)) full(As)\xs
@test_approx_eq solve!(Al,copy(xl)) full(Al)\xl
@printf("OK!\n")

@printf("Symmetric Toeplitz: ")
As = SymmetricToeplitz(0.9.^[0:9])
Al = SymmetricToeplitz(0.9.^[0:999])
@test_approx_eq As*xs full(As)*xs
@test_approx_eq Al*xl full(Al)*xl
@test_approx_eq solve!(As,copy(xs)) full(As)\xs
@test_approx_eq solve!(Al,copy(xl)) full(Al)\xl
@test_approx_eq levinson(As,xs) full(As)\xs
@test_approx_eq levinson(Al,xl) full(Al)\xl

@printf("OK!\n")

@printf("Circulant: ")
As = Circulant(0.9.^[0:9])
Al = Circulant(0.9.^[0:999])
@test_approx_eq As*xs full(As)*xs
@test_approx_eq Al*xl full(Al)*xl
@test_approx_eq solve!(As,copy(xs)) full(As)\xs
@test_approx_eq solve!(Al,copy(xl)) full(Al)\xl
@printf("OK!\n")

@printf("Upper triangular Toeplitz: ")
As = TriangularToeplitz(0.9.^[0:9],:U)
Al = TriangularToeplitz(0.9.^[0:999],:U)
@test_approx_eq As*xs full(As)*xs
@test_approx_eq Al*xl full(Al)*xl
@test_approx_eq solve!(As,copy(xs)) full(As)\xs
@test_approx_eq solve!(Al,copy(xl)) full(Al)\xl
@printf("OK!\n")

@printf("Lower triangular Toeplitz: ")
As = TriangularToeplitz(0.9.^[0:9],:L)
Al = TriangularToeplitz(0.9.^[0:999],:L)
@test_approx_eq As*xs full(As)*xs
@test_approx_eq Al*xl full(Al)*xl
@test_approx_eq solve!(As,copy(xs)) full(As)\xs
@test_approx_eq solve!(Al,copy(xl)) full(Al)\xl
@printf("OK!\n")