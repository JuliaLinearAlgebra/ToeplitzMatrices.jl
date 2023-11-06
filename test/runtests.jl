using Pkg

using ToeplitzMatrices, Test, LinearAlgebra, Aqua, FillArrays, Random
import StatsBase
using FillArrays
using FFTW: fft

@testset "code quality" begin
    Aqua.test_ambiguities(ToeplitzMatrices, recursive=false)
    # Aqua.test_all includes Base and Core in ambiguity testing
    Aqua.test_all(ToeplitzMatrices, ambiguities=false, piracy=false,
        # only test formatting on VERSION >= v1.7
        # https://github.com/JuliaTesting/Aqua.jl/issues/105#issuecomment-1551405866
        project_toml_formatting = VERSION >= v"1.7")
end

ns = 101
nl = 2000

Random.seed!(0)

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
    (UpperTriangularToeplitz(0.9.^(0:ns - 1)),
        UpperTriangularToeplitz(0.9.^(0:nl - 1)),
        "Real upper triangular"),
    (UpperTriangularToeplitz(complex(0.9.^(0:ns - 1))),
        UpperTriangularToeplitz(complex(0.9.^(0:nl - 1))),
        "Complex upper triangular"),
    (LowerTriangularToeplitz(0.9.^(0:ns - 1)),
        LowerTriangularToeplitz(0.9.^(0:nl - 1)),
        "Real lower triangular"),
    (LowerTriangularToeplitz(complex(0.9.^(0:ns - 1))),
         LowerTriangularToeplitz(complex(0.9.^(0:nl - 1))),
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

@testset "vector indexing" begin
    T = Toeplitz(rand(3,3))
    @test T[1:2, 1:2] == Matrix(T)[1:2, 1:2]
    @test AbstractMatrix{ComplexF64}(T) == Toeplitz{ComplexF64}(T.vc, T.vr)
    C = Circulant(1:4)
    @test C[1:2, 1:2] == Matrix(C)[1:2, 1:2]
    @test AbstractMatrix{ComplexF64}(C) == Circulant{ComplexF64}(C.vc)
end

@testset "Mixed types" begin
    @test eltype(Toeplitz([1, 2], [1, 2])) === Int
    @test Toeplitz([1, 2], [1, 2]) * ones(2) == fill(3, 2)
    @test Circulant(Float32.(1:3)) * ones(Float64, 3) == fill(6, 3)
    @test Matrix(Toeplitz(vc, vr)) == Matrix(Toeplitz(vv, vr))
    @test Matrix(Circulant(vc)) == Matrix(Circulant(vv))
    @test Matrix(UpperTriangularToeplitz(vc)) == Matrix(UpperTriangularToeplitz(vv))
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
    @test As' == transpose(As) == As
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

    # testing durbin, levinson, and trench algorithms
    n = 8
    x = range(-1, 1, length = n+1)
    a_exp = @. exp(-abs(x[1]-x))
    a_rbf = @. exp(-abs(x[1]-x)^2) / 2 # this one can be ill-conditioned unless we add a diagonal, i.e. scale kernel down while keeping diagonal at 1
    a_rand = rand(n+1) / n # ensures diagonal dominance
    a_tuple = (a_rand, a_exp, a_rbf)
    for a in a_tuple
        a[1] = 1
        r = a[2:end]
        T = SymmetricToeplitz(vcat(1, r[1:end-1]))
        TM = Matrix(T)

        TS = (2 + randn() / 10) * T # scaled Toeplitz matrix, tests whether non-unit diagonal works
        TSM = Matrix(TS)

        # 1. test durbin algorithm for solves
        y = durbin(r)
        b = - (TM \ r)
        @test b ≈ y
        @inferred durbin(r)

        # 2. test trench algorithm for inversion
        B = trench(r[1:end-1])
        invTM = inv(TM)
        @test B ≈ invTM
        @test trench(T) ≈ invTM
        @test trench(TS) ≈ inv(TSM)
        @inferred trench(TS)

        # 3. test levinson algorithm for solves
        b = randn(n)
        y = levinson(r[1:end-1], b)

        Tb = TM \ b
        @test Tb ≈ y
        @test Tb ≈ levinson(T, b)
        @test TSM \ b ≈ levinson(TS, b)
        @inferred levinson(TS, b)
    end
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

        @test H[1:2, 1:2] == Matrix(H)[1:2, 1:2]
        Hc = AbstractMatrix{ComplexF64}(H)
        @test Hc isa Hankel{ComplexF64}
        @test size(Hc) == size(H)

        @test copy(H) == copyto!(similar(H), H)

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
        @test isa(reverse(T),Hankel)
        @test isa(reverse(T,dims=1),Hankel)
        @test isa(reverse(T,dims=2),Hankel)
    end

    @testset "v too small" begin
        @test_throws ArgumentError Hankel(Int[], (3,4))
        @test_throws ArgumentError Hankel(1:5, (3,4))
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

    T = TriangularToeplitz(ones(2),:L)

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
    @test Toeplitz(A) == Toeplitz{Float64}(A) == A
    @test SymmetricToeplitz(A) == SymmetricToeplitz{Float64}(A) == A
    @test Circulant(A) == Circulant{Float64}(A) == A
    @test Hankel(A) == Hankel{Float64}(A) == A


    A = [1.0 2.0;
         3.0 4.0]

    @test Toeplitz(A) == Toeplitz([1.,3.], [1.,2.]) == Toeplitz{Float64}(A)
    @test SymmetricToeplitz(A) == SymmetricToeplitz{Float64}(A) ==
                Toeplitz(Symmetric(A)) == Symmetric(Toeplitz(A)) == [1. 2.; 2. 1.]
    @test Circulant(A,:L) == [1 3; 3 1] == Circulant(A) == SymmetricToeplitz(A,:L)
    @test Circulant(A,:U) == [1 2; 2 1] == SymmetricToeplitz(A,:U)

    @test TriangularToeplitz(A, :U) == TriangularToeplitz{Float64}(A, :U) == Toeplitz(UpperTriangular(A)) == UpperTriangular(Toeplitz(A)) == UpperTriangularToeplitz(A) == UpperTriangularToeplitz{Float64}(A)
    @test TriangularToeplitz(A, :L) == TriangularToeplitz{Float64}(A, :L) == Toeplitz(LowerTriangular(A)) == LowerTriangular(Toeplitz(A)) == LowerTriangularToeplitz(A) == LowerTriangularToeplitz{Float64}(A)

    @test Hankel(A) == Hankel{Float64}(A) == [1.0 3; 3 4] == Hankel([1.0,3],[3,4]) == Hankel([1.0,3,4],(2,2)) == Hankel([1.0,3,4],2,2) == Hankel{Float64}([1,3,4],(2,2)) == Hankel{Float64}([1.0,3,4],2,2) == Hankel([1.0,3,4])
    @test Hankel(A,:U) == [1.0 2;2 4]

    @test_throws ArgumentError Hankel(A,:🤣)
    @test_throws ArgumentError SymmetricToeplitz(A,:🤣)
    @test_throws ArgumentError Circulant(A,:🤣)
    @test_throws ArgumentError Circulant(1:5,:🤣)
    @test_throws ArgumentError TriangularToeplitz(A,:🤣)

    # Constructors should be projections
    @test Toeplitz(Toeplitz(A)) == Toeplitz(A)
    @test SymmetricToeplitz(SymmetricToeplitz(A)) == SymmetricToeplitz(A)
    @test Circulant(Circulant(A)) == Circulant(A)
    @test TriangularToeplitz(TriangularToeplitz(A, :U), :U) == TriangularToeplitz(A, :U)
    @test TriangularToeplitz(TriangularToeplitz(A, :L), :L) == TriangularToeplitz(A, :L)
    @test Hankel(Hankel(A)) == Hankel(A)

    @test_throws ArgumentError Hankel(1:2,1:2)
    @test_throws ErrorException Toeplitz(1:2,2:1)
end

@testset "General Interface" begin
    for Toep in (:Toeplitz, :Circulant, :SymmetricToeplitz, :UpperTriangularToeplitz, :LowerTriangularToeplitz, :Hankel)
        @eval (A = [1.0 3.0; 3.0 4.0]; TA=$Toep(A); A = Matrix(TA))
        @eval (B = [2   1  ; 1   5  ]; TB=$Toep(B); B = Matrix(TB))

        for fun in (:zero, :conj, :copy, :-, :real, :imag, :adjoint, :transpose, :iszero, :size)
            @eval @test $fun(TA) == $fun(A)
        end

        @test 2*TA == 2*A == lmul!(2,copy(TA))
        @test TA*2 == A*2 == rmul!(copy(TA),2)
        @test TA+TB == A+B
        @test TA-TB == A-B

        @test all(k -> istril(TA, k) == istril(A, k), -5:5)
        @test all(k -> istriu(TA, k) == istriu(A, k), -5:5)

        @test_throws ArgumentError reverse(TA,dims=3)
        if isa(TA,AbstractToeplitz)
            @test isa(reverse(TA),Hankel)
            @test isa(reverse(TA,dims=1),Hankel)
            @test isa(reverse(TA,dims=2),Hankel)
            @test isa(tril(TA),AbstractToeplitz) && tril(TA)==tril(A)
            @test isa(triu(TA),AbstractToeplitz) && triu(TA)==triu(A)
            @test isa(tril(TA,1),AbstractToeplitz) && tril(TA,1)==tril(A,1)
            @test isa(triu(TA,1),AbstractToeplitz) && triu(TA,1)==triu(A,1)
            @test isa(tril(TA,-1),AbstractToeplitz) && tril(TA,-1)==tril(A,-1)
            @test isa(triu(TA,-1),AbstractToeplitz) && triu(TA,-1)==triu(A,-1)
        else
            @test isa(reverse(TA),Toeplitz)
            @test isa(reverse(TA,dims=1),Toeplitz)
            @test isa(reverse(TA,dims=2),Toeplitz)
        end

        T=copy(TA)
        copyto!(T,TB)
        @test T == B

        T=copy(TA)
    end
    @test fill!(Toeplitz(zeros(2,2)),1) == ones(2,2)
	
	@testset "diag" begin
		H = Hankel(1:11, 4, 8)
		@test diag(H) ≡ 1:2:7
		@test diag(H, 1) ≡ 2:2:8
		@test diag(H, -1) ≡ 2:2:6
		@test diag(H, 5) ≡ 6:2:10
		@test diag(H, 100) == diag(H, -100) == []

		T = Toeplitz(1:4, 1:8)
		@test diag(T) ≡ Fill(1, 4)
		@test diag(T, 1) ≡ Fill(2, 4)
		@test diag(T, -1) ≡ Fill(2, 3)
		@test diag(T, 5) ≡ Fill(6, 3)
		@test diag(T, 100) == diag(T, -100) == []
	end

    @testset "istril/istriu/isdiag" begin
        for (vc,vr) in (([1,2,0,0], [1,4,5,0]), ([0,0,0], [0,5,0]), ([3,0,0], [3,0,0]), ([0], [0]))
            for T in (Toeplitz(vc, vr), Circulant(vr),
                            SymmetricToeplitz(vc), LowerTriangularToeplitz(vc),
                            UpperTriangularToeplitz(vr))
                M = Matrix(T)
                for k in -5:5, f in [istriu, istril]
                    @test f(T, k) == f(M, k)
                end
                @test isdiag(T) == isdiag(M)
            end
        end

        for (vr, vc) in (([1,2], [1,2,3,4]), ([1,2,3,4], [1,2]))
            T = Toeplitz(vr, vc)
            M = Matrix(T)
            @testset for f in (istril, istriu)
                @test all(k -> f(T,k) == f(M,k), -5:5)
            end
        end
    end

    @testset "aliasing" begin
        v = [1,2,3]
        T = Toeplitz(v, v)
        @test_throws ArgumentError triu!(T)
        @test_throws ArgumentError tril!(T)
        @test_throws ArgumentError copyto!(T, Toeplitz(1:3, 1:3))
        @test_throws ArgumentError lmul!(2, T)
        @test_throws ArgumentError rmul!(T, 2)

        @test triu(T) == triu(Matrix(T))
        @test tril(T) == tril(Matrix(T))
    end
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

    for t1 in (identity, adjoint), t2 in (identity, adjoint),
            fact1 in (identity, factorize), fact2 in (identity, factorize)
        C = t1(fact1(C1))*t2(fact2(C2))
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
    for (C,M) in ((C1,M1), (C3,M3), (C5,M5))
        λ, V = eigen(C)
        @test C * V ≈ V * Diagonal(λ)
        @test V'V ≈ LinearAlgebra.I
        @test det(C) ≈ det(M)
    end

    # Test for issue #47
    I = inv(C1)*C1
    I2 = inv(factorize(C1))*C1
    e = rand(5)
    # I should be close to identity
    @test I*e ≈ I2*e ≈ e

    D = Diagonal(axes(C1,2))
    @test mul!(similar(C1), C1, D) ≈ C1 * D

    @testset "issymmetric/ishermitian" begin
        C = Circulant([1,2,3,0,3,2])
        @test issymmetric(C)
        @test ishermitian(C)
        C = Circulant([1,2])
        @test issymmetric(C)
        @test ishermitian(C)
        C = Circulant([1,2,3])
        @test !issymmetric(C)
        @test !ishermitian(C)

        C = Circulant([1,im,2-3im,0,2+3im,-im])
        @test ishermitian(C)
        @test !issymmetric(C)
        C = Circulant([1,im])
        @test !ishermitian(C)
        @test issymmetric(C)

        C = Circulant([2])
        @test issymmetric(C)
        @test ishermitian(C)

        C = Circulant([NaN])
        @test !issymmetric(C)
        @test !ishermitian(C)
    end
end

@testset "TriangularToeplitz" begin
    A = [1.0 2.0;
         3.0 4.0]
    TU = UpperTriangularToeplitz(A)
    TL = LowerTriangularToeplitz(A)
    @test (TU * TU)::UpperTriangularToeplitz ≈ Matrix(TU)*Matrix(TU)
    @test (TL * TL)::LowerTriangularToeplitz ≈ Matrix(TL)*Matrix(TL)
    @test (TU * TL) ≈ Matrix(TU)*Matrix(TL)
    for T in (TU, TL)
        @test inv(T)::TriangularToeplitz ≈ inv(Matrix(T))
    end
    A = randn(ComplexF64, 3, 3)
    T = Toeplitz(A)
    TU = UpperTriangular(T)
    @test TU isa TriangularToeplitz
    @test istriu(TU)
    @test TU == Toeplitz(triu(A)) == triu(T)
    @test TU'ones(3) == Matrix(TU)'ones(3)
    @test transpose(TU)*ones(3) == transpose(Matrix(TU))*ones(3)
    @test triu(TU, 1)::TriangularToeplitz == triu(Matrix(T), 1) == triu(T,1)
    TL = LowerTriangular(T)
    @test TL isa TriangularToeplitz
    @test istril(TL)
    @test TL == Toeplitz(tril(A)) == tril(T)
    @test TL'ones(3) == Matrix(TL)'ones(3)
    @test transpose(TL)*ones(3) == transpose(Matrix(TL))*ones(3)
    @test tril(TL, -1)::TriangularToeplitz == tril(Matrix(T), -1) == tril(T,-1)
    for n in (65, 128)
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

@testset "eigen" begin
    @testset "Tridiagonal Toeplitz" begin
        sizes = (1, 2, 5, 6, 10, 15)
        @testset for n in sizes
            if VERSION >= v"1.9"
                @testset "Tridiagonal" begin
                    evm1r = Fill(2, max(0, n-1))
                    ev1r = Fill(3, max(0,n-1))
                    dvr = Fill(-4, n)
                    evm1c = Fill(2+3im, max(0, n-1))
                    dvc = Fill(-4+4im, n)
                    ev1c = Fill(3im, max(0,n-1))

                    for (dl, d, du) in (
                            (evm1r, dvr, ev1r),
                            (evm1r, dvr, -ev1r),
                            (evm1c, dvc, ev1c),
                            )

                        T = Tridiagonal(dl, d, du)
                        λT = eigvals(T)
                        λTM = eigvals(Matrix(T))
                        @test all(x -> any(y -> y ≈ x, λTM), λT)
                        λ, V = eigen(T)
                        @test T * V ≈ V * Diagonal(λ)

                        # Test that internal methods are correct,
                        # aside from the ordering of eigenvectors
                        λT2 = ToeplitzMatrices._eigvals(T)
                        @test all(x -> any(y -> y ≈ x, λTM), λT2)
                        V2 = ToeplitzMatrices._eigvecs(T)
                        for v in eachcol(V2)
                            w = T * v
                            @test any(λ -> w ≈ λ * v, λT2)
                        end
                    end
                end
            end

            @testset "SymTridiagonal/Symmetric" begin
                _dv = Fill(1, n)
                _ev = Fill(3, max(0,n-1))
                for dv in (_dv, -_dv), ev in (_ev, -_ev)
                    for ST in (SymTridiagonal(dv, ev), Symmetric(Tridiagonal(ev, dv, ev)))
                        λST = eigvals(ST)
                        λSTM = eigvals(Matrix(ST))
                        @test all(x -> any(y -> y ≈ x, λSTM), λST)
                        @test eltype(λST) <: Real
                        λ, V = eigen(ST)
                        @test V'V ≈ I
                        @test V' * ST * V ≈ Diagonal(λ)
                    end
                end
                _dv = Fill(-4+4im, n)
                _ev = Fill(2+3im, max(0,n-1))
                for dv in (_dv, -_dv, conj(_dv)), ev in (_ev, -_ev, conj(_ev))
                    for ST2 in (SymTridiagonal(dv, ev), Symmetric(Tridiagonal(ev, dv, ev)))
                        λST = eigvals(ST2)
                        λSTM = eigvals(Matrix(ST2))
                        @test all(x -> any(y -> y ≈ x, λSTM), λST)
                        λ, V = eigen(ST2)
                        @test ST2 * V ≈ V * Diagonal(λ)
                    end
                end
            end

            if VERSION >= v"1.9"
                @testset "Hermitian Tridiagonal" begin
                    _dvR = Fill(2, n)
                    _evR = Fill(3, max(0, n-1))
                    _dvc = complex(_dvR)
                    _evc = Fill(3-4im, max(0, n-1))
                    for (dv, ev) in ((_dvc, _evc), (_dvc, conj(_evc)),
                                        (_dvR, _evR), (_dvR, -_evR))
                        HT = Hermitian(Tridiagonal(ev, dv, ev))
                        λHT = eigvals(HT)
                        λHTM = eigvals(Matrix(HT))
                        @test all(x -> any(y -> y ≈ x, λHTM), λHT)
                        @test eltype(λHT) <: Real
                        λ, V = eigen(HT)
                        @test V'V ≈ I
                        @test V' * HT * V ≈ Diagonal(λ)

                        # Test that internal methods are correct,
                        # aside from the ordering of eigenvectors
                        λHT2 = ToeplitzMatrices._eigvals(HT)
                        @test all(x -> any(y -> y ≈ x, λHTM), λHT2)
                        V2 = ToeplitzMatrices._eigvecs(HT)
                        for v in eachcol(V2)
                            w = HT * v
                            @test any(λ -> w ≈ λ * v, λHT2)
                        end
                    end
                end
            end
        end
    end
end
