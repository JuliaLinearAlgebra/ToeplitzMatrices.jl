using ToeplitzMatrices, StatsBase, Compat, Compat.LinearAlgebra
using Compat.Test

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

        @test As * xs ≈ Matrix(As) * xs
        @test Al * xl ≈ Matrix(Al) * xl
        @test ldiv!(As, Compat.LinearAlgebra.copy_oftype(xs, eltype(As))) ≈ Matrix(As) \ xs
        @test ldiv!(Al, Compat.LinearAlgebra.copy_oftype(xl, eltype(Al))) ≈ Matrix(Al) \ xl
    end
)

@testset "Real general rectangular" begin
    Ar1 = Toeplitz(0.9.^(0:nl-1), 0.4.^(0:ns-1))
    Ar2 = Toeplitz(0.9.^(0:ns-1), 0.4.^(0:nl-1))
    @test Ar1 * xs ≈ Matrix(Ar1) * xs
    @test Ar2 * xl ≈ Matrix(Ar2) * xl
end

@testset "Complex general rectangular" begin
    Ar1 = Toeplitz(complex(0.9.^(0:nl-1)), complex(0.4.^(0:ns-1)))
    Ar2 = Toeplitz(complex(0.9.^(0:ns-1)), complex(0.4.^(0:nl-1)))
    @test Ar1 * xs ≈ Matrix(Ar1) * xs
    @test Ar2 * xl ≈ Matrix(Ar2) * xl
end

@testset "Symmetric Toeplitz" begin
    As = SymmetricToeplitz(0.9.^(0:ns-1))
    Ab = SymmetricToeplitz(abs.(randn(ns)))
    Al = SymmetricToeplitz(0.9.^(0:nl-1))
    @test As * xs ≈ Matrix(As) * xs
    @test Ab * xs ≈ Matrix(Ab) * xs
    @test Al * xl ≈ Matrix(Al) * xl
    @test ldiv!(As, copy(xs)) ≈ Matrix(As) \ xs
    @test ldiv!(Ab, copy(xs)) ≈ Matrix(Ab) \ xs
    @test ldiv!(Al, copy(xl)) ≈ Matrix(Al) \ xl
    @test StatsBase.levinson(As, xs) ≈ Matrix(As) \ xs
    @test StatsBase.levinson(Ab, xs) ≈ Matrix(Ab) \ xs
    if !(haskey(ENV, "CI") && VERSION < v"0.6-") # Inlining is off on 0.5 Travis testing which is too slow for this test
        @test StatsBase.levinson(Al, xl) ≈ Matrix(Al) \ xl
    end
end

@testset "Hankel" begin
    @testset "Real square" begin
        H = Hankel([1.0,2,3,4,5],[5.0,6,7,8,0])
        x = ones(5)
        @test Matrix(H)*x ≈ H*x

        Hs = Hankel(0.9.^(ns-1:-1:0), 0.4.^(0:ns-1))
        Hl = Hankel(0.9.^(nl-1:-1:0), 0.4.^(0:nl-1))
        @test Hs * xs[:,1] ≈ Matrix(Hs) * xs[:,1]
        @test Hs * xs ≈ Matrix(Hs) * xs
        @test Hl * xl ≈ Matrix(Hl) * xl
    end

    @testset "Complex square" begin
        H = Hankel(complex([1.0,2,3,4,5]), complex([5.0,6,7,8,0]))
        x = ones(5)
        @test Matrix(H)*x ≈ H*x

        Hs = Hankel(complex(0.9.^(ns-1:-1:0)), complex(0.4.^(0:ns-1)))
        Hl = Hankel(complex(0.9.^(nl-1:-1:0)), complex(0.4.^(0:nl-1)))
        @test Hs * xs[:,1] ≈ Matrix(Hs) * xs[:,1]
        @test Hs * xs ≈ Matrix(Hs) * xs
        @test Hl * xl ≈ Matrix(Hl) * xl
    end

    @testset "Real rectangular" begin
        Hs = Hankel(0.9.^(ns-1:-1:0), 0.4.^(0:nl-1))
        Hl = Hankel(0.9.^(nl-1:-1:0), 0.4.^(0:ns-1))
        @test Hs * xl[:,1] ≈ Matrix(Hs) * xl[:,1]
        @test Hs * xl ≈ Matrix(Hs) * xl
        @test Hl * xs ≈ Matrix(Hl) * xs
    end

    @testset "Complex rectangular" begin
        Hs = Hankel(complex(0.9.^(ns-1:-1:0)), complex(0.4.^(0:nl-1)))
        Hl = Hankel(complex(0.9.^(nl-1:-1:0)), complex(0.4.^(0:ns-1)))
        @test Hs * xl[:,1] ≈ Matrix(Hs) * xl[:,1]
        @test Hs * xl ≈ Matrix(Hs) * xl
        @test Hl * xs ≈ Matrix(Hl) * xs
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
        @test T*ones(BigFloat,n) ≈ Matrix(T)*ones(BigFloat,n)

        T = TriangularToeplitz(BigFloat[1,2,3,4,5],:L)
        @test T*ones(BigFloat,5) ≈ Matrix(T)*ones(BigFloat,5)

        n = 512
        r = map(BigFloat,rand(n))
        T = TriangularToeplitz(r,:L)
        @test T*ones(BigFloat,n) ≈ Matrix(T)*ones(BigFloat,n)

        T = TriangularToeplitz(BigFloat[1,2,3,4,5],:U)
        @test T*ones(BigFloat,5) ≈ Matrix(T)*ones(BigFloat,5)

        n = 512
        r = map(BigFloat,rand(n))
        T = TriangularToeplitz(r,:U)
        @test T*ones(BigFloat,n) ≈ Matrix(T)*ones(BigFloat,n)
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


    @test Circulant(1:5) == Circulant(Vector(1.0:5))

end


A = ones(10, 10)
@test Matrix(Toeplitz(A)) == Matrix(Toeplitz{Float64}(A)) == A
@test Matrix(SymmetricToeplitz(A)) == Matrix(SymmetricToeplitz{Float64}(A)) == A
@test Matrix(Circulant(A)) == Matrix(Circulant{Float64}(A)) == A
@test Matrix(Hankel(A)) == Matrix(Hankel{Float64}(A)) == A


A = [1.0 2.0;
     3.0 4.0]

@test Toeplitz(A) == Toeplitz([1.,3.], [1.,2.])
@test Toeplitz{Float64}(A) == Toeplitz([1.,3.], [1.,2.])
@test Matrix(SymmetricToeplitz(A)) == Matrix(SymmetricToeplitz{Float64}(A)) ==
            Matrix(Toeplitz(Symmetric(A))) == Matrix(Symmetric(Toeplitz(A))) == [1. 2.; 2. 1.]
@test Matrix(Circulant(A)) == [1 3; 3 1]

@test TriangularToeplitz(A, :U) == TriangularToeplitz{Float64}(A, :U) == Toeplitz(UpperTriangular(A)) == UpperTriangular(Toeplitz(A))
@test TriangularToeplitz(A, :L) == TriangularToeplitz{Float64}(A, :L) == Toeplitz(LowerTriangular(A)) == LowerTriangular(Toeplitz(A))

@test Matrix(Hankel(A)) == Matrix(Hankel{Float64}(A)) == [1.0 3; 3 4]

# Constructors should be projections
@test Toeplitz(Toeplitz(A)) == Toeplitz(A)
@test SymmetricToeplitz(SymmetricToeplitz(A)) == SymmetricToeplitz(A)
@test Circulant(Circulant(A)) == Circulant(A)
@test TriangularToeplitz(TriangularToeplitz(A, :U), :U) == TriangularToeplitz(A, :U)
@test TriangularToeplitz(TriangularToeplitz(A, :L), :L) == TriangularToeplitz(A, :L)
@test Hankel(Hankel(A)) == Hankel(A)
