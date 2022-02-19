using Pkg

# Activate test environment on older Julia versions
@static if VERSION < v"1.2"
    Pkg.activate(@__DIR__)
    Pkg.develop(PackageSpec(; path=dirname(@__DIR__)))
    Pkg.instantiate()
end

using ToeplitzMatrices, StatsBase, Test, LinearAlgebra

using FFTW: fft

ns = 101
nl = 2000

xs = randn(ns, 5)
xl = randn(nl, 5)
vc = 1.0:3.0 # for testing with AbstractVector
vv = Vector(vc)
vr = [1, 5.]

cases = [
    (Toeplitz(0.9.^(0:ns-1), 0.4.^(0:ns-1)),
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
         "Complex lower triangular"),
]

for (As, Al, st) in cases
    @testset "Toeplitz: $st" begin
        @test As * xs ≈ Matrix(As)  * xs
        @test As'* xs ≈ Matrix(As)' * xs
        @test Al * xl ≈ Matrix(Al)  * xl
        @test Al'* xl ≈ Matrix(Al)' * xl
        @test [As[n] for n in 1:length(As)] == vec(As)
        @test [Al[n] for n in 1:length(Al)] == vec(Al)
        @test ldiv!(As, LinearAlgebra.copy_oftype(xs, eltype(As))) ≈ Matrix(As) \ xs
        @test ldiv!(Al, LinearAlgebra.copy_oftype(xl, eltype(Al))) ≈ Matrix(Al) \ xl
        @test Matrix(As') == Matrix(As)'
        @test Matrix(transpose(As)) == transpose(Matrix(As))
    end
end

@testset "Mixed types" begin
    @test eltype(Toeplitz([1, 2], [1, 2])) === Int
    @test Toeplitz([1, 2], [1, 2]) * ones(2) == fill(3, 2)
    @test Circulant(Float32.(1:3)) * ones(Float64, 3) == fill(6, 3)
    @test Matrix(Toeplitz(vc, vr)) == Matrix(Toeplitz(vv, vr))
    @test Matrix(Circulant(vc)) == Matrix(Circulant(vv))
    @test Matrix(TriangularToeplitz(vc,:U)) == Matrix(TriangularToeplitz(vv,:U))
end

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
    @test Matrix(SymmetricToeplitz(vc)) == Matrix(SymmetricToeplitz(vv))
end

@testset "Hankel" begin
    @testset "Real square" begin
        H = Hankel([1.0,2,3,4,5],[5.0,6,7,8,9])
        @test Matrix(H) == [1 2 3 4 5;
                            2 3 4 5 6;
                            3 4 5 6 7;
                            4 5 6 7 8;
                            5 6 7 8 9]

        @test convert(Hankel{Float64}, H) == H
        @test convert(AbstractMatrix{Float64}, H) == H
        @test convert(AbstractArray{Float64}, H) == H

        @test H[2,2] == 3
        @test H[7]  == 3
        @test diag(H) == [1,3,5,7,9]

        x = ones(5)
        @test mul!(copy(x), H, x) ≈ Matrix(H)*x ≈ H*x

        Hs = Hankel(0.9.^(ns-1:-1:0), 0.4.^(0:ns-1))
        Hl = Hankel(0.9.^(nl-1:-1:0), 0.4.^(0:nl-1))
        @test Hs * xs[:,1] ≈ Matrix(Hs) * xs[:,1]
        @test mul!(copy(xs), Hs, xs) ≈ Hs * xs ≈ Matrix(Hs) * xs
        @test mul!(copy(xl), Hl, xl) ≈ Hl * xl ≈ Matrix(Hl) * xl
        @test Matrix(Hankel(reverse(vc),vr)) == Matrix(Hankel(reverse(vv),vr))
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

    @testset "Convert" begin
        H = Hankel([1.0,2,3,4,5],[5.0,6,7,8,0])
        @test Hankel(H) == Hankel{Float64}(H) == H
        @test convert(Hankel,H) == convert(Hankel{Float64},H) ==
                convert(AbstractArray,H) == convert(AbstractArray{Float64},H) == H

        A = [1.0 2; 3 4]
        @test Hankel(A) == [1 3; 3 4]
        T = Toeplitz([1.0,2,3,4,5],[1.0,6,7,8,0])
        @test Hankel(T) == Hankel([1.0,2,3,4,5],[5.0,4,3,2,1])
        @test Hankel(T) ≠ ToeplitzMatrices._Hankel(T)
    end
end

@testset "Convert" begin
    T = Toeplitz(ones(2),ones(2))

    @test isa(convert(Matrix{ComplexF64},T),Matrix{ComplexF64})
    @test isa(convert(AbstractMatrix{ComplexF64},T),Toeplitz{ComplexF64})
    @test isa(convert(AbstractArray{ComplexF64},T),Toeplitz{ComplexF64})
    @test isa(convert(ToeplitzMatrices.AbstractToeplitz{ComplexF64},T),Toeplitz{ComplexF64})
    @test isa(convert(ToeplitzMatrices.Toeplitz{ComplexF64},T),Toeplitz{ComplexF64})

    T = SymmetricToeplitz(ones(2))

    @test isa(convert(Matrix{Float32},T),Matrix{Float32})
    @test isa(convert(AbstractMatrix{Float32},T),SymmetricToeplitz{Float32})
    @test isa(convert(AbstractArray{Float32},T),SymmetricToeplitz{Float32})
    @test isa(convert(ToeplitzMatrices.AbstractToeplitz{Float32},T),SymmetricToeplitz{Float32})
    @test isa(convert(ToeplitzMatrices.SymmetricToeplitz{Float32},T),SymmetricToeplitz{Float32})

    T = Circulant(ones(2))

    @test isa(convert(Matrix{ComplexF64},T),Matrix{ComplexF64})
    @test isa(convert(AbstractMatrix{ComplexF64},T),Circulant{ComplexF64})
    @test isa(convert(AbstractArray{ComplexF64},T),Circulant{ComplexF64})
    @test isa(convert(ToeplitzMatrices.AbstractToeplitz{ComplexF64},T),Circulant{ComplexF64})
    @test isa(convert(ToeplitzMatrices.Circulant{ComplexF64},T),Circulant{ComplexF64})

    T = TriangularToeplitz(ones(2),:U)

    @test isa(convert(Matrix{ComplexF64},T),Matrix{ComplexF64})
    @test isa(convert(AbstractMatrix{ComplexF64},T),TriangularToeplitz{ComplexF64})
    @test isa(convert(AbstractArray{ComplexF64},T),TriangularToeplitz{ComplexF64})
    @test isa(convert(ToeplitzMatrices.AbstractToeplitz{ComplexF64},T),TriangularToeplitz{ComplexF64})
    @test isa(convert(ToeplitzMatrices.TriangularToeplitz{ComplexF64},T),TriangularToeplitz{ComplexF64})
    @test isa(convert(Toeplitz, T), Toeplitz)

    T = Hankel(ones(2),ones(2))

    @test isa(convert(Matrix{ComplexF64},T),Matrix{ComplexF64})
    @test isa(convert(AbstractMatrix{ComplexF64},T),Hankel{ComplexF64})
    @test isa(convert(AbstractArray{ComplexF64},T),Hankel{ComplexF64})
    @test isa(convert(ToeplitzMatrices.Hankel{ComplexF64},T),Hankel{ComplexF64})


    @test Circulant(1:5) == Circulant(Vector(1.0:5))

end

@testset "Constructors" begin
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
end

@testset "Circulant mathematics" begin
    C1 = Circulant(rand(5))
    C2 = Circulant(rand(5))
    C3 = Circulant{ComplexF64}(rand(5))
    C4 = Circulant(ones(5))
    C5 = Circulant{ComplexF64}(ones(5))
    M1 = Matrix(C1)
    M2 = Matrix(C2)
    M3 = Matrix(C3)
    M4 = Matrix(C4)
    M5 = Matrix(C5)

    for t1 in (identity, adjoint), t2 in (identity, adjoint), fact in (identity, factorize)
        C = t1(fact(C1))*t2(fact(C2))
        @test C isa Circulant
        @test C ≈ t1(M1)*t2(M2)
    end

    C = C1-C2
    @test C isa Circulant
    @test C ≈ M1-M2

    C = C1+C2
    @test C isa Circulant
    @test C ≈ M1+M2

    C = 2C1
    @test C isa Circulant
    @test C ≈ 2M1

    C = C1*2
    @test C isa Circulant
    @test C ≈ M1*2

    C = -C1
    @test C isa Circulant
    @test C ≈ -M1

    C = inv(C1)
    @test C isa Circulant
    @test C ≈ inv(M1)
    @test fft(C.vc) ≈ (factorize(C)).vcvr_dft
    C = inv(C3)
    @test C isa Circulant
    @test C ≈ inv(M3)
    @test fft(C.vc) ≈ (factorize(C)).vcvr_dft

    C = pinv(C1)
    @test C isa Circulant
    @test C ≈ pinv(M1)
    @test fft(C.vc) ≈ (factorize(C)).vcvr_dft
    C = pinv(C3)
    @test C isa Circulant
    @test C ≈ pinv(M3)
    @test fft(C.vc) ≈ (factorize(C)).vcvr_dft
    C = pinv(C4)
    @test C isa Circulant
    @test C ≈ pinv(M4)
    @test fft(C.vc) ≈ (factorize(C)).vcvr_dft
    C = pinv(C5)
    @test C isa Circulant
    @test C ≈ pinv(M5)
    @test fft(C.vc) ≈ (factorize(C)).vcvr_dft

    C = sqrt(C1)
    @test C isa Circulant
    @test C*C ≈ C1
    C = sqrt(C3)
    @test C isa Circulant
    @test C*C ≈ C3

    C = copy(C1)
    @test C isa Circulant
    C2 = similar(C1)
    copyto!(C2, C1)
    @test C1 ≈ C2

    v1 = eigvals(C1)
    v2 = eigvals(M1)
    for v1i in v1
        @test minimum(abs.(v1i .- v2)) < sqrt(eps(Float64))
    end

    # Test for issue #47
    I = inv(C1)*C1
    I2 = inv(factorize(C1))*C1
    e = rand(5)
    # I should be close to identity
    @test I*e ≈ I2*e ≈ e
end

@testset "TriangularToeplitz" begin
    A = [1.0 2.0;
         3.0 4.0]
    TU = TriangularToeplitz(A, :U)
    TL = TriangularToeplitz(A, :L)
    @test (TU * TU)::TriangularToeplitz ≈ Matrix(TU)*Matrix(TU)
    @test (TL * TL)::TriangularToeplitz ≈ Matrix(TL)*Matrix(TL)
    @test (TU * TL) ≈ Matrix(TU)*Matrix(TL)
    for T in (TU, TL)
        @test inv(T)::TriangularToeplitz ≈ inv(Matrix(T))
    end
    for n in (65, 128)
        @show n
        A = randn(n, n)
        TU = TriangularToeplitz(A, :U)
        TL = TriangularToeplitz(A, :L)
        @test_broken inv(TU)::TriangularToeplitz ≈ inv(Matrix(TU))
        @test inv(TL)::TriangularToeplitz ≈ inv(Matrix(TL))
    end
end

@testset "Cholesky" begin
    T = SymmetricToeplitz(exp.(-0.5 .* range(0, stop=5, length=100)))
    @test cholesky(T).U ≈ cholesky(Matrix(T)).U
    @test cholesky(T).L ≈ cholesky(Matrix(T)).L
end
