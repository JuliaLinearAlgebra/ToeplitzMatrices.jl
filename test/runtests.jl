using Pkg

# Activate test environment on older Julia versions
@static if VERSION < v"1.2"
    Pkg.activate(@__DIR__)
    Pkg.develop(PackageSpec(; path=dirname(@__DIR__)))
    Pkg.instantiate()
end

using ToeplitzMatrices, StatsBase, Test, LinearAlgebra

using Base: copyto!
using FFTW: fft

const atol = 1e-6
const rtol = 1e-6
function isapprox_helper(x, y, verbose::Bool = true)
    b = isapprox(x, y, atol = atol, rtol = rtol)
    if ~b && verbose
        println("x !≈ y")
        println("norm(x-y): ", norm(x-y))
        println("norm(x-y)/norm(y): ", norm(x-y)/norm(y))
    end
    return b
end

ns = 17
nl = 33
k = 1

xs = randn(ns, k)
vc = LinRange(1, 3, 3) # for testing with AbstractVector
vv = Vector(vc)
vr = [1, 5.]

diag_val = 2 # elevating diagonal encourages well-conditioned spectrum
sizes = (17, 33, 513, 1024)
for n in sizes
    @testset "n = $n" begin
        exp_95 = 0.9.^(0:n-1)
        exp_40 = 0.4.^(0:n-1)
        exp_40[1] = exp_95[1] = diag_val

        rs1 = randn(n) / n # this scaling encourages good conditioning and diagonal dominance
        rs2 = randn(n) / n
        rs1[1] = rs2[1] = diag_val #  making sure the first elements have the same value

        cases = [
            (Toeplitz(exp_95, exp_40),
                "Real square"),
            (Toeplitz(complex.(exp_95), complex.(exp_40)),
                "Complex square"),
            (Toeplitz(rs1, rs2),
                "Real random square"),
            (Toeplitz(complex(rs1), complex(rs2)),
                "Complex random square"),
            (Circulant(exp_95),
                "Real circulant"),
            (Circulant(complex.(exp_95)),
                "Complex circulant"),
            (Circulant(rs1),
                "Real random circulant"),
            (Circulant(complex.(rs1)),
                "Complex random circulant"),
            (TriangularToeplitz(exp_95, :U),
                "Real upper triangular"),
            (TriangularToeplitz(complex.(exp_95), :U),
                "Complex upper triangular"),
            (TriangularToeplitz(exp_95, :L),
                "Real lower triangular"),
            (TriangularToeplitz(complex.(exp_95), :L),
                 "Complex lower triangular"),
        ]

        for (A, st) in cases
            x = randn(eltype(A), n)
            X = randn(eltype(A), n, k) # for multiplication and solver tests
            @testset "Toeplitz: $st" begin
                M = Matrix(A)
                @test A * x ≈ M  * x
                @test A'* x ≈ M' * x
                @test A * X ≈ M  * X
                @test A'* X ≈ M' * X
                @test [A[n] for n in 1:length(A)] == vec(A)
                @test isapprox_helper(A \ x, M \ x)
                @test isapprox_helper(A \ X, M \ X)
                @test isapprox_helper(ldiv!(A, LinearAlgebra.copy_oftype(X, eltype(A))), M \ X)
                @test Matrix(A') == M'
                @test adjoint(factorize(A)) * X ≈ M' * X
                @test Matrix(transpose(A)) == transpose(M)
                @test size(A) == size(M)
                @test size(A, 3) == size(M, 3)
                F = factorize(A)
                @test size(F) == size(M)
                @test size(F, 3) == size(M, 3)
                @test A * A ≈ M * M
                @test isapprox_helper(inv(A), inv(M))
            end # testset matrices
        end # loop over matrices

        @testset "Rectangular" begin
            for m in sizes
                @testset "m = $m" begin
                    exp_40_m = 0.4.^(0:m-1)
                    exp_40_m[1] = diag_val
                    rectangular_cases = [
                        (Toeplitz(exp_95, exp_40_m),
                            "Real general rectangular"),
                        (Toeplitz(complex.(exp_95), complex.(exp_40_m)),
                            "Complex general rectangular"),
                    ]

                    for (A, st) in rectangular_cases
                        xn, xm = randn(eltype(A), n, k), randn(eltype(A), m, k)
                        @testset "Toeplitz: $st" begin
                            M = Matrix(A)
                            @test A * xm ≈ M  * xm
                            @test A'* xn ≈ M' * xn
                            @test [A[n] for n in 1:length(A)] == vec(A)
                            @test isapprox(A \ xn, M \ xn, atol = 1e-4, rtol = 1e-4)
                            @test Matrix(A') == M'
                            @test Matrix(transpose(A)) == transpose(M)
                        end
                    end
                end # testset m
            end # loop m
        end

        @testset "Symmetric Toeplitz" begin
            Xs = randn(n, k)
            xs = randn(n)

            As = SymmetricToeplitz(exp_95)
            Ms = Matrix(As)
            @test isapprox_helper(As * xs, Ms * xs)
            @test isapprox_helper(As * Xs, Ms * Xs)
            @test isapprox_helper(ldiv!(As, copy(xs)), Ms \ xs)
            @test isapprox_helper(ldiv!(As, copy(Xs)), Ms \ Xs)
            @test StatsBase.levinson(As, xs) ≈ Ms \ xs # this should be exact to numerical precision given good conditioning
            @test Matrix(As') ≈ Ms'
            @test Matrix(transpose(As)) ≈ transpose(Ms)
            @test ldiv!(zero(xs), As, xs, isposdef = true) ≈ As \ xs # test cg-based solve

            Ab = SymmetricToeplitz(rs1) # this is still positive definite
            Mb = Matrix(Ab)
            @test isapprox_helper(Ab * xs, Mb * xs)
            @test isapprox_helper(Ab * Xs, Mb * Xs)
            @test isapprox_helper(ldiv!(Ab, copy(xs)), Mb \ xs)
            @test isapprox_helper(ldiv!(Ab, copy(Xs)), Mb \ Xs)
            @test StatsBase.levinson(Ab, xs) ≈ Mb \ xs
            @test Matrix(Ab') ≈ Mb'
            @test Matrix(transpose(Ab)) ≈ transpose(Mb)

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
                @test Matrix(H)*x ≈ H*x

                Hs = Hankel(0.9.^(n-1:-1:0), 0.4.^(0:n-1))
                xs = randn(n)
                @test Hs * xs[:,1] ≈ Matrix(Hs) * xs[:,1]
                @test Hs * xs ≈ Matrix(Hs) * xs
                @test Matrix(Hankel(reverse(vc),vr)) == Matrix(Hankel(reverse(vv),vr))
            end

            @testset "Complex square" begin
                H = Hankel(complex([1.0,2,3,4,5]), complex([5.0,6,7,8,0]))
                x = ones(5)
                @test Matrix(H)*x ≈ H*x

                Hs = Hankel(complex(0.9.^(n-1:-1:0)), complex(0.4.^(0:n-1)))
                xs = randn(n)
                @test Hs * xs[:,1] ≈ Matrix(Hs) * xs[:,1]
                @test Hs * xs ≈ Matrix(Hs) * xs
            end

            for m in sizes
                xm = randn(m)
                exp_95_n = reverse(exp_95)
                exp_40_m = 0.4.^(0:m-1)
                exp_40_m[1] = diag_val
                @testset "m = $m" begin
                    @testset "Real rectangular" begin
                        Hs = Hankel(exp_95_n, exp_40_m)
                        @test Hs * xm[:,1] ≈ Matrix(Hs) * xm[:,1]
                        @test Hs * xm ≈ Matrix(Hs) * xm
                    end

                    @testset "Complex rectangular" begin
                        Hs = Hankel(complex.(exp_95_n), complex.(exp_40_m))
                        @test Hs * xm[:,1] ≈ Matrix(Hs) * xm[:,1]
                        @test Hs * xm ≈ Matrix(Hs) * xm
                    end
                end # testset m
            end # loop m

            @testset "Convert" begin
                H = Hankel([1.0,2,3,4,5],[5.0,6,7,8,0])
                @test Hankel(H) == Hankel{Float64}(H) == H
                @test convert(Hankel,H) == convert(Hankel{Float64},H) ==
                        convert(AbstractArray,H) == convert(AbstractArray{Float64},H) == H
                @test convert(Array, H) ≈ Matrix(H)

                A = [1.0 2; 3 4]
                @test Hankel(A) == [1 3; 3 4]
                T = Toeplitz([1.0,2,3,4,5],[1.0,6,7,8,0])
                @test Hankel(T) == Hankel([1.0,2,3,4,5],[5.0,4,3,2,1])
                @test Hankel(T) ≠ ToeplitzMatrices._Hankel(T)
            end
        end

    end # testset n
end # loop over n

@testset "Mixed types" begin
    @test eltype(Toeplitz([1, 2], [1, 2])) === Int
    @test Toeplitz([1, 2], [1, 2]) * ones(2) == fill(3, 2)
    @test Circulant(Float32.(1:3)) * ones(Float64, 3) == fill(6, 3)
    @test Matrix(Toeplitz(vc, vr)) == Matrix(Toeplitz(vv, vr))
    @test Matrix(Circulant(vc)) == Matrix(Circulant(vv))
    @test Matrix(TriangularToeplitz(vc,:U)) == Matrix(TriangularToeplitz(vv,:U))
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

    T = TriangularToeplitz(ones(2), :U)

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

    C = C1*C2
    @test C isa Circulant
    @test C ≈ M1*M2

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

    lambda_abs = eigvals(abs(C1))
    abs_lambda = abs.(eigvals(C1))
    @test all(<(1e-12), imag(lambda_abs))
    lambda_abs = real.(lambda_abs)
    @test all(>(0), lambda_abs)
    @test sort!(lambda_abs) ≈ sort!(abs_lambda)

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
    e = rand(5)
    # I should be close to identity
    @test I*e ≈ e
end

@testset "Cholesky" begin
    T = SymmetricToeplitz(exp.(-0.5 .* range(0, stop=5, length=100)))
    @test cholesky(T).U ≈ cholesky(Matrix(T)).U
    @test cholesky(T).L ≈ cholesky(Matrix(T)).L
end
