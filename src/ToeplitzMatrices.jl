module ToeplitzMatrices

using IterativeLinearSolvers

import Base: full, getindex, print_matrix, size, tril, triu, *, inv, A_mul_B!, Ac_mul_B, A_ldiv_B!
import Base.LinAlg: BlasFloat, BlasReal, DimensionMismatch

export Toeplitz, SymmetricToeplitz, Circulant, TriangularToeplitz, 
	   chan, strang, A_mul_B!, Ac_mul_B!, levinson

solve(A::AbstractMatrix, b::AbstractVector) = A_ldiv_B!(zeros(length(b)), A, b)
A_ldiv_B!(x::AbstractVector, A::AbstractMatrix, b::AbstractVector) = x[:] = A\b

# Abstract
abstract AbstractToeplitz{T<:Number} <: AbstractMatrix{T}

size(A::AbstractToeplitz) = (size(A, 1), size(A, 2))
getindex(A::AbstractToeplitz, i::Integer) = A[mod(i, size(A,1)), div(i, size(A,1)) + 1]
function full{T}(A::AbstractToeplitz{T})
	m, n = size(A)
	Af = Array(T, m, n)
	for j = 1:n
		for i = 1:m
			Af[i,j] = A[i,j]
		end
	end
	return Af
end

function A_mul_B!{T<:BlasFloat}(α::T, A::AbstractToeplitz{T}, x::StridedVector{T}, β::T, y::AbstractVector{T})
	n = length(y)
	n2 = length(A.vc_dft)
	if n != size(A,1) throw(DimensionMismatch("")) end
	if n != length(x) throw(DimensionMismatch("")) end
	if n < 512
		y[:] *= β
		for j = 1:n
			tmp = α*x[j]
			for i = 1:n
				y[i] += tmp*A[i,j]
			end
		end
		return y
	end
	for i = 1:n
		A.tmp[i] = complex(x[i])
	end
	for i = n+1:n2
		A.tmp[i] = zero(Complex{T})
	end
	A.dft(A.tmp)
	for i = 1:n2
		A.tmp[i] *= A.vc_dft[i]
	end
	A.idft(A.tmp)
	for i = 1:n
		y[i] *= β
		y[i] += α*real(A.tmp[i])
	end
	return y
end
(*){T}(A::AbstractToeplitz{T},x::AbstractVector{T}) = A_mul_B!(one(T),A,x,zero(T),zeros(T,length(x)))

function strang{T}(A::AbstractMatrix{T})
	n = size(A,1)
	v = Array(T, n)
	n2 = div(n,2)
	for i = 1:n
		if i <= n2 + 1
			v[i] = A[i,1]
		else
			v[i] = A[1,n-i+2]
		end
	end
	return Circulant(v)
end
function chan{T}(A::AbstractMatrix{T})
	n = size(A,1)
	v = Array(T,n)
	for i = 1:n
		v[i] = ((n-i+1)*A[i,1] + (i-1)*A[1,min(n-i+2,n)])/n
	end
	return Circulant(v)
end

# General
type Toeplitz{T<:Number} <: AbstractToeplitz{T}
	vc::Vector{T}
	vr::Vector{T}
	vc_dft::Vector{Complex{T}}
	tmp::Vector{Complex{T}}
	dft::Function
	idft::Function
end
function Toeplitz{T<:BlasReal}(vc::Vector{T}, vr::Vector{T})
	n = length(vc)
	if length(vr) != n throw(DimensionMismatch("")) end
	if vc[1] != vr[1] error("First element of the vectors must be the same") end
	tmp = complex([vc, zero(T), reverse(vr[2:end])])
	dft = plan_fft!(tmp)
	return Toeplitz(vc, vr, dft(tmp), similar(tmp), dft, plan_ifft!(tmp))
end

size(A::Toeplitz, dim::Int) = dim == 1 ? length(A.vc) : (dim == 2 ? length(A.vr) : (dim > 2 ? 1 : error("arraysize: dimension out of range")))

function getindex(A::Toeplitz, i::Integer, j::Integer)
	n = size(A,1)
	if i > n || j > n error("BoundsError()") end
	tmp = i - j
	if tmp >= 0 return A.vc[tmp+1] end
	return A.vr[1-tmp]
end

function tril{T}(A::Toeplitz{T}, k::Integer)
	if k > 0 error("Second argument cannot be positive") end
	Al = TriangularToeplitz(copy(A.vc), 'L')
	for i in -1:-1:k
		Al.ve[-i] = zero(T)
	end
	return Al
end
function triu{T}(A::Toeplitz{T}, k::Integer)
	if k < 0 error("Second argument cannot be negative") end
	Al = TriangularToeplitz(copy(A.vr), 'U')
	for i in 1:k
		Al.ve[i] = zero(T)
	end
	return Al
end


# *(A::Toeplitz, B::VecOrMat) = tril(A)*B + triu(A, 1)*B
(*)(A::Toeplitz, b::Vector) = irfft(rfft([A.vc, reverse(A.vr[2:end])]).*rfft([b,zeros(length(b)-1)]),2length(b) - 1)[1:length(b)]

A_ldiv_B!(A::Toeplitz, b::StridedVector) = cgs(A,zeros(length(b)),b,strang(A),1000,100eps())[1]

# Symmetric
type SymmetricToeplitz{T<:BlasFloat} <: AbstractToeplitz{T}
	vc::Vector{T}
	vc_dft::Vector{Complex{T}}
	tmp::Vector{Complex{T}}
	dft::Function
	idft::Function
end
function SymmetricToeplitz{T<:BlasFloat}(vc::Vector{T})
	tmp = convert(Array{Complex{T}}, [vc, zero(T), reverse(vc[2:end])])
	dft = plan_fft!(tmp)
	return SymmetricToeplitz(vc, dft(tmp), similar(tmp), dft, plan_ifft!(tmp))
end

size(A::SymmetricToeplitz, dim::Int) = 1 <= dim <=2 ? length(A.vc)  : error("arraysize: dimension out of range")

getindex(A::SymmetricToeplitz, i::Integer, j::Integer) = A.vc[abs(i-j)+1]

(*){T<:BlasFloat}(A::SymmetricToeplitz{T}, x::Vector{T}) = A_mul_B!(one(T),A,x,zero(T),zeros(T, length(x)))

function durbin!{T<:BlasFloat}(r::AbstractVector{T}, y::AbstractVector{T})
	n = length(r)
	if n != length(y) throw(DimensionMismatch("Vector must have same length")) end
	y[1] = -r[1]
	β = one(T)
	α = -r[1]
	for k = 1:n-1
		β *= one(T) - α*α
		α = -r[k+1]
		for j = 1:k
			α -= r[k-j+2]*y[j]
		end
		α /= β
		for j = 1:div(k,2)
			tmp = y[j]
			y[j] += α*y[k-j+1]
			y[k-j+1] += α*tmp
		end
		if isodd(k) y[div(k,2)+1] *= one(T) + α end
		y[k+1] = α
	end
	return y
end
durbin(r::AbstractVector) = durbin!(r, zeros(length(r)))

function levinson!{T<:BlasFloat}(r::AbstractVector{T}, b::AbstractVector{T}, x::AbstractVector{T})
	n = length(b)
	if n != length(r) throw(DimensionMismatch("")) end
	x[1] = b[1]
	b[1] = -r[2]/r[1]
	β = one(T)
	α = -r[2]/r[1]
	for k = 1:n-1
		β *= one(T) - α*α
		μ = b[k+1]
		for j = 2:k+1
			μ -= r[j]/r[1]*x[k-j+2]
		end
		μ /= β
		for j = 1:k
			x[j] += μ*b[k-j+1]
		end
		x[k+1] = μ
		if k < n - 1
			α = -r[k+2]
			for j = 2:k+1
				α -= r[j]*b[k-j+2]
			end
			α /= β*r[1]
			for j = 1:div(k,2)
				tmp = b[j]
				b[j] += α*b[k-j+1]
				b[k-j+1] += α*tmp
			end
			if isodd(k) b[div(k,2)+1] *= one(T) + α end
			b[k+1] = α
		end
	end
	for i = 1:n
		x[i] /= r[1]
	end
	return x
end
levinson!(x::AbstractVector, A::SymmetricToeplitz, b::AbstractVector) = levinson!(A.vc, b, x)
levinson(r::AbstractVector, b::AbstractVector) = levinson!(r, copy(b), zeros(length(b)))
levinson(A::AbstractToeplitz, b::AbstractVector) = levinson!(A, copy(b), zeros(length(b)))

A_ldiv_B!(A::SymmetricToeplitz,b::StridedVector) = cg(A,zeros(length(b)),b,strang(A),1000,100eps())[1]

# Circulant
type Circulant{T<:BlasReal} <: AbstractToeplitz{T}
	vc::Vector{T}
	vc_dft::Vector{Complex{T}}
	tmp::Vector{Complex{T}}
	dft::Function
	idft::Function
end
function Circulant{T<:BlasReal}(vc::Vector{T})
	tmp = zeros(Complex{T}, length(vc))
	return Circulant(vc, fft(vc), tmp, plan_fft!(tmp), plan_ifft!(tmp))
end

size(C::Circulant, dim::Integer) = 1 <= dim <=2 ? length(C.vc) : error("arraysize: dimension out of range")

function getindex(C::Circulant, i::Integer, j::Integer)
	n = size(C, 1)
	if i > n || j > n error("BoundsError()") end
	return C.vc[mod(i-j, length(C.vc))+1]
end

function Ac_mul_B(A::Circulant,B::Circulant)
	tmp = similar(A.vc_dft)
	for i = 1:length(tmp)
		tmp[i] = conj(A.vc_dft[i])*B.vc_dft[i]
	end
	return Circulant(real(A.idft(tmp)), tmp, A.tmp, A.dft, A.idft)
end

function A_ldiv_B!{T<:BlasReal}(C::Circulant{T}, b::AbstractVector{T})
	n = length(b)
	size(C, 1) == n || throw(DimensionMismatch(""))
	for i = 1:n
		C.tmp[i] = b[i]
	end
	C.dft(C.tmp)
	for i = 1:n
		C.tmp[i] /= C.vc_dft[i]
	end
	C.idft(C.tmp)
	for i = 1:n
		b[i] = real(C.tmp[i])
	end
	return b
end
(\)(C::Circulant, b::AbstractVector) = A_ldiv_B!(C, copy(b))

function inv{T<:BlasReal}(C::Circulant{T})
	vdft = 1/C.vc_dft
	return Circulant(real(C.idft(vdft)), copy(vdft), similar(vdft), C.dft, C.idft)
end

# Triangular
type TriangularToeplitz{T<:Number} <: AbstractToeplitz{T}
	ve::Vector{T}
	uplo::Char
	vc_dft::Vector{Complex{T}}
	tmp::Vector{Complex{T}}
	dft::Function
	idft::Function
end
function TriangularToeplitz{T<:BlasReal}(ve::Vector{T}, uplo::Symbol)
	n = length(ve)
	tmp = uplo == :L ? complex([ve, zeros(n)]) : complex([ve[1], zeros(T, n), reverse(ve[2:end])])
	dft = plan_fft!(tmp)
	return TriangularToeplitz(ve, string(uplo)[1], dft(tmp), similar(tmp), dft, plan_ifft!(tmp))
end

size(A::TriangularToeplitz, dim::Int) = (dim == 1) | (dim == 2) ? length(A.ve) : (dim > 2 ? 1 : error("arraysize: dimension out of range"))

function getindex{T}(A::TriangularToeplitz{T}, i::Integer, j::Integer)
	if A.uplo == 'L'
		return i >= j ? A.ve[i-j+1] : zero(T)
	else
		return i <= j ? A.ve[j-i+1] : zero(T)
	end
end

function *(A::TriangularToeplitz, B::TriangularToeplitz)
	n = size(A, 1)
	if n != size(B, 1) throw(DimensionMismatch("")) end
	if A.uplo == B.uplo
		return TriangularToeplitz(conv(A.ve, B.ve)[1:n], A.uplo)
	end
	return Triangular(full(A), A.uplo)*Triangular(full(B), B.uplo)
end

Ac_mul_B(A::TriangularToeplitz, b::AbstractVector) = *(TriangularToeplitz(A.ve, A.uplo == 'U' ? :L : :U), b)

# NB! only valid for loser trianggular
function smallinv{T<:BlasFloat}(A::TriangularToeplitz{T})
	n = size(A, 1)
	b = zeros(T, n)
	b[1] = 1/A.ve[1]
	for k = 2:n
		tmp = zero(T)
		for i = 1:k-1
			tmp += A.uplo == 'L' ? A.ve[k-i+1]*b[i] : A.ve[i+1]*b[k-i]
		end
		b[k] = -tmp/A.ve[1]
	end
	return TriangularToeplitz(b, symbol(A.uplo))
end

function inv{T}(A::TriangularToeplitz{T})
	n = size(A, 1)
	if n <= 64 return smallinv(A) end
	np2 = nextpow2(n)
	if n != np2 return TriangularToeplitz(inv(TriangularToeplitz([A.ve, zeros(T, np2 - n)], symbol(A.uplo))).ve[1:n], symbol(A.uplo)) end
	nd2 = div(n, 2)
	a1 = inv(TriangularToeplitz(A.ve[1:nd2], symbol(A.uplo))).ve
	return TriangularToeplitz([a1, -(TriangularToeplitz(a1, symbol(A.uplo))*(Toeplitz(A.ve[nd2+1:end], A.ve[nd2+1:-1:2])*a1))], symbol(A.uplo))
end

# A_ldiv_B!(A::TriangularToeplitz,b::StridedVector) = inv(A)*b
A_ldiv_B!(A::TriangularToeplitz,b::StridedVector) = cgs(A,zeros(length(b)),b,chan(A),1000,100eps())[1]

# BlockTriangular
# type BlockTriangularToeplitz{T<:BlasReal} <: AbstractMatrix{T}
# 	Mc::Array{T,3}
# 	uplo::Char
# 	Mc_dft::Array{Complex{T},3}
# 	tmp::Vector{Complex{T}}
# 	dft::Function
# 	idft::Function
# end
# function BlockTriangularToeplitz{T<:BlasReal}(Mc::Array{T,3}, uplo::Symbol)
# 	n, p, _ = size(Mc)
# 	tmp = zeros(Complex{T}, 2n)
# 	dft = plan_fft!(tmp)
# 	idft = plan_ifft!(tmp)
# 	Mc_dft = Array(Complex{T}, 2n, p, p)
# 	for j = 1:p
# 		for i = 1:p
# 			Mc_dft[1,i,j] = complex(Mc[1,i,j])
# 			for t = 2:n
# 				Mc_dft[t,i,j] = uplo == :L ? complex(Mc[t,i,j]) : zero(Complex{T})
# 			end
# 			Mc_dft[n+1,i,j] = zero(Complex{T})
# 			for t = n+2:2n
# 				Mc_dft[t,i,j] = uplo == :L ? zero(Complex{T}) : complex(Mc[2n-t+2,i,j])
# 			end
# 			dft(sub(Mc_dft, 1:2n, 1:p, 1:p))
# 		end
# 	end
# 	return BlockTriangularToeplitz(Mc, string(uplo)[1], Mc_dft, tmp, dft, idft)
end