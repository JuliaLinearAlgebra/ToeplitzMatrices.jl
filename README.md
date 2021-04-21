ToeplitzMatrices.jl
===========

[![Build Status](https://github.com/JuliaMatrices/ToeplitzMatrices.jl/workflows/CI/badge.svg?branch=master)](https://github.com/JuliaMatrices/ToeplitzMatrices.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/JuliaMatrices/ToeplitzMatrices.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaMatrices/ToeplitzMatrices.jl)
[![Coverage](https://coveralls.io/repos/github/JuliaMatrices/ToeplitzMatrices.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaMatrices/ToeplitzMatrices.jl?branch=master)

Fast matrix multiplication and division
for Toeplitz, Hankel and circulant matrices in Julia

# Note

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

# Supported matrices

## Toeplitz

A Toeplitz matrix has constant diagonals. It can be constructed using

```julia
Toeplitz(vc,vr)
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

## SymmetricToeplitz

A symmetric Toeplitz matrix is a symmetric matrix with constant diagonals. It can be constructed with

```julia
SymmetricToeplitz(vc)
```

where `vc` are the entries of the first column. For example,

```julia
SymmetricToeplitz([1.0, 2.0, 3.0])
```

is a sparse representation of the matrix

```julia
[ 1.0  2.0  3.0
  2.0  1.0  2.0
  3.0  2.0  1.0 ]
```

## TriangularToeplitz

A triangular Toeplitz matrix can be constructed using

```julia
TriangularToeplitz(ve,uplo)
```

where uplo is either `:L` or `:U` and `ve` are the rows or columns, respectively.  For example,

```julia
TriangularToeplitz([1.,2.,3.],:L)
```

is a sparse representation of the matrix

```julia
[ 1.0  0.0  0.0
  2.0  1.0  0.0
  3.0  2.0  1.0 ]
```

## Hankel

A Hankel matrix has constant anti-diagonals.  It can be constructed using

```julia
Hankel(vc,vr)
```

where `vc` are the entries in the first column and `vr` are the entries in the last row, where `vc[end]` must equal `vr[1]`.  For example.

```julia
Hankel([1.,2.,3.], 3:5)
```

is a sparse representation of the matrix

```julia
[  1.0  2.0  3.0
   2.0  3.0  4.0
   3.0  4.0  5.0 ]
```

## Circulant

A circulant matrix is a special case of a Toeplitz matrix with periodic end conditions.
It can be constructed using

```julia
Circulant(vc)
```
where `vc` is a vector with the entries for the first column.
For example:
```julia
Circulant([1.0, 2.0, 3.0])
```
is a sparse representation of the matrix

```julia
[  1.0  3.0  2.0
   2.0  1.0  3.0
   3.0  2.0  1.0 ]
```
