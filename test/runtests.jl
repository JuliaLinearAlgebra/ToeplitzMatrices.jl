using Base.Test
using ToeplitzMatrices

ns = 101
nl = 2000

xs = randn(ns, 5)
xl = randn(nl, 5)

println("Toeplitz")
for (As, Al, st) in ((Toeplitz(0.9.^(0:ns-1), 0.4.^(0:ns-1)),
                        Toeplitz(0.9.^(0:nl-1), 0.4.^(0:nl-1)),
                            "Real general square"),
                     (Toeplitz(complex(0.9.^(0:ns-1)), complex(0.4.^(0:ns-1))),
                        Toeplitz(complex(0.9.^(0:nl-1)), complex(0.4.^(0:nl-1))),
                            "Complex general square"),
                     (Circulant(0.9.^(0:ns - 1)),
                        Circulant(0.9.^(0:nl - 1)),
                            "Real circulant"),
                     (Circulant(complex(0.9.^(0:ns - 1))),
                        Circulant(complex(0.9.^(0:nl - 1))),
                            "Complex circulant"),
                     (TriangularToeplitz(0.9.^(0:ns - 1), :U),
                        TriangularToeplitz(0.9.^(0:nl - 1), :U),
                            "Real upper triangular"),
                     (TriangularToeplitz(complex(0.9.^(0:ns - 1)), :U),
                        TriangularToeplitz(complex(0.9.^(0:nl - 1)), :U),
                            "Complex upper triangular"),
                     (TriangularToeplitz(0.9.^(0:ns - 1), :L),
                        TriangularToeplitz(0.9.^(0:nl - 1), :L),
                            "Real lower triangular"),
                     (TriangularToeplitz(complex(0.9.^(0:ns - 1)), :L),
                        TriangularToeplitz(complex(0.9.^(0:nl - 1)), :L),
                            "Complex lower triangular"))
    print("$st: ")
    @test_approx_eq As * xs full(As) * xs
    @test_approx_eq Al * xl full(Al) * xl
    @test_approx_eq A_ldiv_B!(As, LinAlg.copy_oftype(xs, eltype(As))) full(As) \ xs
    @test_approx_eq A_ldiv_B!(Al, LinAlg.copy_oftype(xl, eltype(Al))) full(Al) \ xl
    println("OK!")
end

print("Real general rectangular: ")
Ar1 = Toeplitz(0.9.^(0:nl-1), 0.4.^(0:ns-1))
Ar2 = Toeplitz(0.9.^(0:ns-1), 0.4.^(0:nl-1))
@test_approx_eq Ar1 * xs full(Ar1) * xs
@test_approx_eq Ar2 * xl full(Ar2) * xl
println("OK!")

print("Complex general rectangular: ")
Ar1 = Toeplitz(complex(0.9.^(0:nl-1)), complex(0.4.^(0:ns-1)))
Ar2 = Toeplitz(complex(0.9.^(0:ns-1)), complex(0.4.^(0:nl-1)))
@test_approx_eq Ar1 * xs full(Ar1) * xs
@test_approx_eq Ar2 * xl full(Ar2) * xl
println("OK!")

print("Symmetric Toeplitz: ")
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
println("OK!")

println("\nHankel")
print("Real square: ")
H = Hankel([1.0,2,3,4,5],[5.0,6,7,8,0])
x = ones(5)
@test_approx_eq full(H)*x H*x

Hs = Hankel(0.9.^(ns-1:-1:0), 0.4.^(0:ns-1))
Hl = Hankel(0.9.^(nl-1:-1:0), 0.4.^(0:nl-1))
@test_approx_eq Hs * xs[:,1] full(Hs) * xs[:,1]
@test_approx_eq Hs * xs full(Hs) * xs
@test_approx_eq Hl * xl full(Hl) * xl
println("OK!")

print("Complex square: ")
H = Hankel(complex([1.0,2,3,4,5]), complex([5.0,6,7,8,0]))
x = ones(5)
@test_approx_eq full(H)*x H*x

Hs = Hankel(complex(0.9.^(ns-1:-1:0)), complex(0.4.^(0:ns-1)))
Hl = Hankel(complex(0.9.^(nl-1:-1:0)), complex(0.4.^(0:nl-1)))
@test_approx_eq Hs * xs[:,1] full(Hs) * xs[:,1]
@test_approx_eq Hs * xs full(Hs) * xs
@test_approx_eq Hl * xl full(Hl) * xl
println("OK!")

print("Real rectangular: ")
Hs = Hankel(0.9.^(ns-1:-1:0), 0.4.^(0:nl-1))
Hl = Hankel(0.9.^(nl-1:-1:0), 0.4.^(0:ns-1))
@test_approx_eq Hs * xl[:,1] full(Hs) * xl[:,1]
@test_approx_eq Hs * xl full(Hs) * xl
@test_approx_eq Hl * xs full(Hl) * xs
println("OK!")

print("Complex rectangular: ")
Hs = Hankel(complex(0.9.^(ns-1:-1:0)), complex(0.4.^(0:nl-1)))
Hl = Hankel(complex(0.9.^(nl-1:-1:0)), complex(0.4.^(0:ns-1)))
@test_approx_eq Hs * xl[:,1] full(Hs) * xl[:,1]
@test_approx_eq Hs * xl full(Hs) * xl
@test_approx_eq Hl * xs full(Hl) * xs
println("OK!")


if isdir(Pkg.dir("FastTransforms"))
    print("\nBigFloat")
    using FastTransforms
    T = Toeplitz(BigFloat[1,2,3,4,5], BigFloat[1,6,7,8,0])
    @test_approx_eq T*ones(BigFloat,5) [22,24,19,16,15]

    n = 512
    r = map(BigFloat,rand(n))
    T = Toeplitz(r,[r[1];map(BigFloat,rand(n-1))])
    @test_approx_eq T*ones(BigFloat,n) full(T)*ones(BigFloat,n)

    T = TriangularToeplitz(BigFloat[1,2,3,4,5],:L)
    @test_approx_eq T*ones(BigFloat,5) full(T)*ones(BigFloat,5)

    n = 512
    r = map(BigFloat,rand(n))
    T = TriangularToeplitz(r,:L)
    @test_approx_eq T*ones(BigFloat,n) full(T)*ones(BigFloat,n)

    T = TriangularToeplitz(BigFloat[1,2,3,4,5],:U)
    @test_approx_eq T*ones(BigFloat,5) full(T)*ones(BigFloat,5)

    n = 512
    r = map(BigFloat,rand(n))
    T = TriangularToeplitz(r,:U)
    @test_approx_eq T*ones(BigFloat,n) full(T)*ones(BigFloat,n)
    println("OK!")
end
