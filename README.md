ToeplitzMatrices.jl
===========

[![Build Status](https://travis-ci.org/JuliaMatrices/ToeplitzMatrices.jl.svg?branch=master)](https://travis-ci.org/JuliaMatrices/ToeplitzMatrices.jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaMatrices/ToeplitzMatrices.jl/badge.svg?branch=master&bust=1)](https://coveralls.io/github/JuliaMatrices/ToeplitzMatrices.jl?branch=master)

Fast matrix multiplication and division for Toeplitz and Hankel matrices in Julia


## ToeplitzMatrix

A Toeplitz matrix has constant diagonals.  It can be constructed using

```julia
Toeplitz(vc,vr)
```

where `vc` are the entries in the first column and `vr` are the entries in the first row, where `vc[1]` must equal `vr[1]`.  For example.

```julia
Toeplitz([1.,2.,3.],[1.,4.,5.])
```

is a sparse representation of the matrix

```julia
[ 1.0  4.0  5.0
  2.0  1.0  4.0
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

 # Hankel

 A Hankel matrix has constant anti-diagonals.  It can be constructed using

 ```julia
 Hankel(vc,vr)
 ```

 where `vc` are the entries in the first column and `vr` are the entries in the last row, where `vc[end]` must equal `vr[1]`.  For example.

 ```julia
 Hankel([1.,2.,3.],[3.,4.,5.])
 ```

 is a sparse representation of the matrix

 ```julia
 [  1.0  2.0  3.0
    2.0  3.0  4.0
    3.0  4.0  5.0 ]
 ```


 # Circulant
 
 A circulant matrix is a special case of a Toeplitz matrix with periodic end conditions.
 It can be constructed using
 
 ```julia
 Circulant(vc)
 ```
where `vc` is a vector with the entries for the first column.
For example:
```julia
 Circulant(1:3)
 ```
 is a sparse representation of the matrix

 ```julia
 [  1.0  3.0  2.0
    2.0  1.0  3.0
    3.0  2.0  1.0 ]
 ```
