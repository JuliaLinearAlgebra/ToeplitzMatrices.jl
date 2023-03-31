ToeplitzMatrices.jl
===========

[![Build Status](https://github.com/JuliaLinearAlgebra/ToeplitzMatrices.jl/workflows/CI/badge.svg?branch=master)](https://github.com/JuliaLinearAlgebra/ToeplitzMatrices.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/JuliaLinearAlgebra/ToeplitzMatrices.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaLinearAlgebra/ToeplitzMatrices.jl)
[![Coverage](https://coveralls.io/repos/github/JuliaLinearAlgebra/ToeplitzMatrices.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaLinearAlgebra/ToeplitzMatrices.jl?branch=master)

Fast matrix multiplication and division
for Toeplitz, Hankel and circulant matrices in Julia

## Note

Multiplication of large matrices and `sqrt`, `inv`, `LinearAlgebra.eigvals`,
`LinearAlgebra.ldiv!`, and `LinearAlgebra.pinv` for circulant matrices
are computed with FFTs.
To be able to use these methods, you have to install and load a package that implements
the [AbstractFFTs.jl](https://github.com/JuliaMath/AbstractFFTs.jl) interface such
as [FFTW.jl](https://github.com/JuliaMath/FFTW.jl):

```julia
using FFTW
```

If you perform multiple calculations with FFTs, it can be more efficient to
initialize the required arrays and plan the FFT only once. You can precompute
the FFT factorization with `LinearAlgebra.factorize` and then use the factorization
for the FFT-based computations.

## Introduction

### Toeplitz
A Toeplitz matrix has constant diagonals. It can be constructed using
```julia
Toeplitz(vc,vr)
Toeplitz{T}(vc,vr)
```
where `vc` are the entries in the first column and `vr` are the entries in the first row, where `vc[1]` must equal `vr[1]`. For example.
```julia
Toeplitz(1:3, [1.,4.,5.])
```
is a sparse representation of the matrix
```julia
[ 1.0  4.0  5.0
  2.0  1.0  4.0
  3.0  2.0  1.0 ]
```
### Special toeplitz
`SymmetricToeplitz`, `Circulant`, `UpperTriangularToeplitz` and `LowerTriangularToeplitz` only store one vector. By convention, `Circulant` stores the first column rather than the first row. They are constructed using `TYPE(v)` where `TYPE`∈{`SymmetricToeplitz`, `Circulant`, `UpperTriangularToeplitz`, `LowerTriangularToeplitz`}.

### Hankel
A Hankel matrix has constant anti-diagonals, for example,
```julia
[ 1  2  3
  2  3  4 ]
```
There are a few ways to construct the above `Hankel{Int}`:
- `Hankel([1,2,3,4], (2,3)) # Hankel(v, (h,w))`
- `Hankel([1,2,3,4], 2,3) # Hankel(v, h, w)`
- `Hankel([1,2], [2,3,4]) # Hankel(vc, vr)`

Note that the width is usually useless, since ideally, `w=length(v)-h+1`. It exists for infinite Hankel matrices. Its existence also means that `v` could be longer than necessary. `Hankel(v)`, where the size is not given, returns `Hankel(v, (l+1)÷2, (l+1)÷2)` where `l=length(v)`.

The `reverse` can transform between Hankel and Toeplitz. It is used to achieve fast Hankel algorithms.

## Implemented interface

### Operations

<details>
  <summary>Full list</summary>
  
- ✓ implemented
- ✗ error
- _ fall back to `Matrix`

||Toeplitz|Symmetric~|Circulant|UpperTriangular~|LowerTriangular~|Hankel|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|getindex|✓|✓|✓|✓|✓|✓|
|.vc|✓|✓|✓|✓|✓|✓|
|.vr|✓|✓|✓|✓|✓|✓|
|size|✓|✓|✓|✓|✓|✓|
|copy|✓|✓|✓|✓|✓|✓|
|similar|✓|✓|✓|✓|✓|✓|
|zero|✓|✓|✓|✓|✓|✓|
|real|✓|✓|✓|✓|✓|✓|
|imag|✓|✓|✓|✓|✓|✓|
|fill!|✓|✗|✗|✗|✗|✓|
|conj|✓|✓|✓|✓|✓|✓|
|transpose|✓|✓|✓|✓|✓|✓|
|adjoint|✓|✓|✓|✓|✓|✓|
|tril!|✓|✗|✗|✓|✓|✗|
|triu!|✓|✗|✗|✓|✓|✗|
|tril|✓|✓|✓|✓|✓|✗|
|triu|✓|✓|✓|✓|✓|✗|
|+|✓|✓|✓|✓|✓|✓|
|-|✓|✓|✓|✓|✓|✓|
|scalar<br>mult|✓|✓|✓|✓|✓|✓|
|==|✓|✓|✓|✓|✓|✓|
|issymmetric|||||||
|istriu|||||||
|istril|||||||
|iszero|✓|✓|✓|✓|✓||
|isone|||||||
|diag|✓|✓|✓|✓|✓|✓|
|copyto!|✓|✓|✓|✓|✓|✓|
|reverse|✓|✓|✓|✓|✓|✓|
|broadcast|||||||
|broadcast!|||||||
  
</details>

Note that scalar multiplication, `conj`, `+` and `-` could be removed once `broadcast` is implemented.

`reverse(Hankel)` returns a `Toeplitz`, while `reverse(AbstractToeplitz)` returns a `Hankel`.

### LinearAlgebra

### Constructors and conversions
||Toeplitz|Symmetric~|Circulant|UpperTriangular~|LowerTriangular~|Hankel|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|from AbstractVector|✓|✓|✓|✓|✓|✓|
|from AbstractMatrix|✓|✓|✓|✓|✓|✓|
|from AbstractToeplitz|✓|✓|✓|✓|✓|✗|
|to supertype|✓|✓|✓|✓|✓|✓|
|to Toeplitz|-|✓|✓|✓|✓|✗|
|to another eltype|✓|✓|✓|✓|✓|✓|

When constructing `Toeplitz` from a matrix, the first row and the first column will be considered as `vr` and `vc`. Note that `vr` and `vc` are copied in construction to avoid the cases where they share memory. If you don't want copying, construct using vectors directly.

When constructing `SymmetricToeplitz` or `Circulant` from `AbstractMatrix`, a second argument shall specify whether the first row or the first column is used. For example, for `A = [1 2; 3 4]`, 
- `SymmetricToeplitz(A,:L)` gives `[1 3; 3 1]`, while
- `SymmetricToeplitz(A,:U)` gives `[1 2; 2 1]`.

For backward compatibility and consistency with `LinearAlgebra.Symmetric`,
```julia
SymmetricToeplitz(A) = SymmetricToeplitz(A, :U)
Circulant(A) = Circulant(A, :L)
```
`Hankel` constructor also accepts the second argument, `:L` denoting the first column and the last row while `:U` denoting the first row and the last column.

`Symmetric`, `UpperTriangular` and `LowerTriangular` from `LinearAlgebra` are also overloaded for convenience.
```julia
Symmetric(T::Toeplitz) = SymmetricToeplitz(T)
UpperTriangular(T::Toeplitz) = UpperTriangularToeplitz(T)
LowerTriangular(T::Toeplitz) = LowerTriangularToeplitz(T)
```

### TriangularToeplitz (obsolete)
`TriangularToeplitz` is reserved for backward compatibility. 
```julia
TriangularToeplitz = Union{UpperTriangularToeplitz,LowerTriangularToeplitz}
```
The old interface is implemented by
```julia
getproperty(UpperTriangularToeplitz,:uplo) = :U
getproperty(LowerTriangularToeplitz,:uplo) = :L
```
This type is **obsolete** and will not be updated for features. Despite that, backward compatibility should be maintained. Codes that were using `TriangularToeplitz` should still work.

## Unexported interface
Methods in this section are not exported.

`_vr(A::AbstractMatrix)` returns the first row as a vector.
`_vc(A::AbstractMatrix)` returns the first column as a vector.
`_vr` and `_vc` are implemented for `AbstractToeplitz` as well. They are used to merge similar codes for `AbstractMatrix` and `AbstractToeplitz`.

`_circulate(v::AbstractVector)` converts between the `vr` and `vc` of a `Circulant`.

`isconcrete(A::Union{AbstractToeplitz,Hankel})` decides whether the stored vector(s) are concrete. It calls `Base.isconcretetype`.
