if VERSION ≤ v"0.7.0-DEV.1775"
    using Base.Test
else
    using Test
end
using ToeplitzMatrices, StatsBase

ns = 101
nl = 2000

xs = randn(ns, 5)
xl = randn(nl, 5)

@testset("Toeplitz: $st",
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

    @test As * xs ≈ full(As) * xs
    @test Al * xl ≈ full(Al) * xl
    @test A_ldiv_B!(As, LinAlg.copy_oftype(xs, eltype(As))) ≈ full(As) \ xs
    @test A_ldiv_B!(Al, LinAlg.copy_oftype(xl, eltype(Al))) ≈ full(Al) \ xl
end)

@testset "Real general rectangular" begin
    Ar1 = Toeplitz(0.9.^(0:nl-1), 0.4.^(0:ns-1))
    Ar2 = Toeplitz(0.9.^(0:ns-1), 0.4.^(0:nl-1))
    @test Ar1 * xs ≈ full(Ar1) * xs
    @test Ar2 * xl ≈ full(Ar2) * xl
end

@testset "Complex general rectangular" begin
    Ar1 = Toeplitz(complex(0.9.^(0:nl-1)), complex(0.4.^(0:ns-1)))
    Ar2 = Toeplitz(complex(0.9.^(0:ns-1)), complex(0.4.^(0:nl-1)))
    @test Ar1 * xs ≈ full(Ar1) * xs
    @test Ar2 * xl ≈ full(Ar2) * xl
end

@testset "Symmetric Toeplitz" begin
    As = SymmetricToeplitz(0.9.^(0:ns-1))
    Ab = SymmetricToeplitz(abs.(randn(ns)))
    Al = SymmetricToeplitz(0.9.^(0:nl-1))
    @test As * xs ≈ full(As) * xs
    @test Ab * xs ≈ full(Ab) * xs
    @test Al * xl ≈ full(Al) * xl
    @test A_ldiv_B!(As, copy(xs)) ≈ full(As) \ xs
    @test A_ldiv_B!(Ab, copy(xs)) ≈ full(Ab) \ xs
    @test A_ldiv_B!(Al, copy(xl)) ≈ full(Al) \ xl
    @test StatsBase.levinson(As, xs) ≈ full(As) \ xs
    @test StatsBase.levinson(Ab, xs) ≈ full(Ab) \ xs
    if !(haskey(ENV, "CI") && VERSION < v"0.6-") # Inlining is off on 0.5 Travis testing which is too slow for this test
        @test StatsBase.levinson(Al, xl) ≈ full(Al) \ xl
    end
end

@testset "Hankel" begin
    @testset "Real square" begin
        H = Hankel([1.0,2,3,4,5],[5.0,6,7,8,0])
        x = ones(5)
        @test full(H)*x ≈ H*x

        Hs = Hankel(0.9.^(ns-1:-1:0), 0.4.^(0:ns-1))
        Hl = Hankel(0.9.^(nl-1:-1:0), 0.4.^(0:nl-1))
        @test Hs * xs[:,1] ≈ full(Hs) * xs[:,1]
        @test Hs * xs ≈ full(Hs) * xs
        @test Hl * xl ≈ full(Hl) * xl
    end

    @testset "Complex square" begin
        H = Hankel(complex([1.0,2,3,4,5]), complex([5.0,6,7,8,0]))
        x = ones(5)
        @test full(H)*x ≈ H*x

        Hs = Hankel(complex(0.9.^(ns-1:-1:0)), complex(0.4.^(0:ns-1)))
        Hl = Hankel(complex(0.9.^(nl-1:-1:0)), complex(0.4.^(0:nl-1)))
        @test Hs * xs[:,1] ≈ full(Hs) * xs[:,1]
        @test Hs * xs ≈ full(Hs) * xs
        @test Hl * xl ≈ full(Hl) * xl
    end

    @testset "Real rectangular" begin
        Hs = Hankel(0.9.^(ns-1:-1:0), 0.4.^(0:nl-1))
        Hl = Hankel(0.9.^(nl-1:-1:0), 0.4.^(0:ns-1))
        @test Hs * xl[:,1] ≈ full(Hs) * xl[:,1]
        @test Hs * xl ≈ full(Hs) * xl
        @test Hl * xs ≈ full(Hl) * xs
    end

    @testset "Complex rectangular" begin
        Hs = Hankel(complex(0.9.^(ns-1:-1:0)), complex(0.4.^(0:nl-1)))
        Hl = Hankel(complex(0.9.^(nl-1:-1:0)), complex(0.4.^(0:ns-1)))
        @test Hs * xl[:,1] ≈ full(Hs) * xl[:,1]
        @test Hs * xl ≈ full(Hs) * xl
        @test Hl * xs ≈ full(Hl) * xs
    end
end

if isdir(Pkg.dir("FastTransforms"))
    using FastTransforms
end
if isdir(Pkg.dir("FastTransforms"))
    @testset "BigFloat" begin
        T = Toeplitz(BigFloat[1,2,3,4,5], BigFloat[1,6,7,8,0])
        @test T*ones(BigFloat,5) ≈ [22,24,19,16,15]

        n = 512
        r = map(BigFloat,rand(n))
        T = Toeplitz(r,[r[1];map(BigFloat,rand(n-1))])
        @test T*ones(BigFloat,n) ≈ full(T)*ones(BigFloat,n)

        T = TriangularToeplitz(BigFloat[1,2,3,4,5],:L)
        @test T*ones(BigFloat,5) ≈ full(T)*ones(BigFloat,5)

        n = 512
        r = map(BigFloat,rand(n))
        T = TriangularToeplitz(r,:L)
        @test T*ones(BigFloat,n) ≈ full(T)*ones(BigFloat,n)

        T = TriangularToeplitz(BigFloat[1,2,3,4,5],:U)
        @test T*ones(BigFloat,5) ≈ full(T)*ones(BigFloat,5)

        n = 512
        r = map(BigFloat,rand(n))
        T = TriangularToeplitz(r,:U)
        @test T*ones(BigFloat,n) ≈ full(T)*ones(BigFloat,n)
    end
end


@testset "Convert" begin

    T = Toeplitz(ones(2),ones(2))

    @test isa(convert(Matrix{Complex128},T),Matrix{Complex128})
    @test isa(convert(AbstractMatrix{Complex128},T),Toeplitz{Complex128})
    @test isa(convert(AbstractArray{Complex128},T),Toeplitz{Complex128})
    @test isa(convert(ToeplitzMatrices.AbstractToeplitz{Complex128},T),Toeplitz{Complex128})
    @test isa(convert(ToeplitzMatrices.Toeplitz{Complex128},T),Toeplitz{Complex128})

    T = SymmetricToeplitz(ones(2))

    @test isa(convert(Matrix{Float32},T),Matrix{Float32})
    @test isa(convert(AbstractMatrix{Float32},T),SymmetricToeplitz{Float32})
    @test isa(convert(AbstractArray{Float32},T),SymmetricToeplitz{Float32})
    @test isa(convert(ToeplitzMatrices.AbstractToeplitz{Float32},T),SymmetricToeplitz{Float32})
    @test isa(convert(ToeplitzMatrices.SymmetricToeplitz{Float32},T),SymmetricToeplitz{Float32})

    T = Circulant(ones(2))

    @test isa(convert(Matrix{Complex128},T),Matrix{Complex128})
    @test isa(convert(AbstractMatrix{Complex128},T),Circulant{Complex128})
    @test isa(convert(AbstractArray{Complex128},T),Circulant{Complex128})
    @test isa(convert(ToeplitzMatrices.AbstractToeplitz{Complex128},T),Circulant{Complex128})
    @test isa(convert(ToeplitzMatrices.Circulant{Complex128},T),Circulant{Complex128})

    T = TriangularToeplitz(ones(2),:U)

    @test isa(convert(Matrix{Complex128},T),Matrix{Complex128})
    @test isa(convert(AbstractMatrix{Complex128},T),TriangularToeplitz{Complex128})
    @test isa(convert(AbstractArray{Complex128},T),TriangularToeplitz{Complex128})
    @test isa(convert(ToeplitzMatrices.AbstractToeplitz{Complex128},T),TriangularToeplitz{Complex128})
    @test isa(convert(ToeplitzMatrices.TriangularToeplitz{Complex128},T),TriangularToeplitz{Complex128})

    T = Hankel(ones(2),ones(2))

    @test isa(convert(Matrix{Complex128},T),Matrix{Complex128})
    @test isa(convert(AbstractMatrix{Complex128},T),Hankel{Complex128})
    @test isa(convert(AbstractArray{Complex128},T),Hankel{Complex128})
    @test isa(convert(ToeplitzMatrices.Hankel{Complex128},T),Hankel{Complex128})
end
