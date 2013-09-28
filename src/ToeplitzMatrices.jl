module ToeplitzMatrices

import Base: full, getindex, print_matrix, size, tril, triu, *, inv, A_mul_B, Ac_mul_B
import Base.LinAlg: BlasFloat, BlasReal, DimensionMismatch

export Toeplitz, SymmetricToeplitz, Circulant, TriangularToeplitz, 
	   chan, strang, A_mul_B!, Ac_mul_B!, solve!

solve(A::AbstractMatrix, b::AbstractVector) = solve!(zeros(length(b)), A, b)
solve!(x::AbstractVector, A::AbstractMatrix, b::AbstractVector) = x[:] = A\b

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

# Symmetric
type SymmetricToeplitz{T<:BlasFloat} <: AbstractToeplitz{T}
	vc::Vector{T}
	vc_dft::Vector{T}
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

*{T<:BlasFloat}(A::SymmetricToeplitz{T}, x::Vector{T}) = A_mul_B!(one(T),A,x,zero(T),zeros(T, length(x)))

function solve_levinson!{T<:BlasFloat}(x::AbstractVector{T}, r::AbstractVector{T}, b::AbstractVector{T})
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
solve_levinson!(x::AbstractVector, A::SymmetricToeplitz, b::AbstractVector) = solve_levinson!(x, A.vc, b)
solve_levinson(r::AbstractVector, b::AbstractVector) = solve_levinson!(zeros(length(b)), r, copy(b))
solve_levinson(A::AbstractMatrix, b::AbstractVector) = solve_levinson!(zeros(length(b)), A, copy(b))

function solve!(x::AbstractVector, A::SymmetricToeplitz, b::AbstractVector, method::Symbol=:Levinson)
	if method == :Levinson return solve_levinson!(x,A,b) end
	if method == :CG return solve_cg!(x,A,b) end
	error("No such method")
end
# Curculant
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

function solve!{T<:BlasReal}(C::Circulant{T}, b::AbstractVector{T})
	n = length(b)
	if size(C, 1) != n throw(DimensionMismatch("")) end
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

inv{T<:BlasFloat}(C::Circulant{T}) = Circulant(convert(Vector{T}, ifft(1/fft(C.vc))))

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
	tmp = uplo == :L ? complex([ve, zeros(n)]) : complex([ve[1], zeros(T, n - 1), ve[2:end]])
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
	if A.uplo == 'U' error("Not implemented yet") end
	n = size(A, 1)
	b = zeros(T, n)
	b[1] = 1/A.ve[1]
	for k = 2:n
		tmp = zero(T)
		for i = 1:k-1
			tmp += A.ve[k-i+1]*b[i]
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

function inv2{T}(A::TriangularToeplitz{T})
	n = size(A, 1)
	if n <= 64 return smallinv(A) end
	np2 = nextpow2(n)
	if n != np2 return TriangularToeplitz(inv2(TriangularToeplitz([A.ve, zeros(T, np2 - n)], symbol(A.uplo))).ve[1:n], symbol(A.uplo)) end
	nd2 = div(n, 2)
	a = TriangularToeplitz(A.ve[1:nd2], A.uplo)
	ainv = inv2(a)
	while LinAlg.BLAS.asum(a.ve)*LinAlg.BLAS.asum(ainv.ve) > 1e9
		a.ve[1] += eps()
		a = TriangularToeplitz(A.ve[1:nd2], A.uplo)
		ainv = inv2(a)
	end
	return TriangularToeplitz([ainv.ve, -(ainv*(Toeplitz(A.ve[nd2+1:end], A.ve[nd2+1:-1:2])*ainv.ve))], A.uplo)
end

# function solve_cg!(A::TriangularToeplitz, b::Vector, x::Vector, maxiter::Int, xtol::Float64)
# 	r0 = A'b - A'*(A*x)
# 	p = r0
# 	for i = 1:maxiter
# 		Ap = A*p
# 		a = dot(r0,r0)/dot(Ap,Ap)
# 		x += a*p
# 		r1 = r0 - a*(A'Ap)
# 		if norm(r1) < xtol break end
# 		b = dot(r1,r1)/dot(r0,r0)
# 		p = r1 + b*p
# 		r0 = copy(r1)
# 		# println(i)
# 	end
# 	return x
# end

# function solve_cg2!{T<:BlasFloat}(x::Vector{T}, A::TriangularToeplitz{T}, b::AbstractVector{T}, maxiter::Int=100, xtol::T=length(b)*eps(typeof(b[1])), p = Array(T,length(b)), Ap = zeros(T,length(b)))
# 	n = length(b)
# 	rr0 = one(T)
# 	rr1 = one(T)
# 	xtolnb = xtol*norm(b)

# 	k = 0
# 	A_mul_B!(-one(T),A,x,one(T),b)
# 	while norm(b) > xtolnb && k < maxiter
# 		k += 1
# 		if k == 1
# 			Ac_mul_B!(one(T),A,b,zero(T),p)
# 			rr1 = one(T)
# 		else
# 			Ac_mul_B!(one(T),A,b,zero(T),Ap)
# 			rr1 = dot(Ap,Ap)
# 			β = rr1/rr0
# 			p[:] *= β
# 			p[:] += Ap
# 		end
# 		A_mul_B!(one(T),A,p,zero(T),Ap)
# 		α = rr1/dot(Ap,Ap)
# 		for i = 1:n
# 			x[i] += α*p[i]
# 			b[i] -= α*Ap[i]
# 		end
# 		rr0 = rr1
# 	end
# 	return x, k
# end
# solve_cg2(A::TriangularToeplitz, b::AbstractVector) = solve_cg2!(zeros(length(b)), A, copy(b))

# function solve_pcg!{T<:BlasFloat}(x::Vector{T}, A::TriangularToeplitz{T}, b::AbstractVector{T}, M::AbstractMatrix{T}, maxiter::Int=100, xtol::T=length(b)*eps(typeof(b[1])), p = zeros(T,length(b)), Ap = zeros(T,length(b)), z = zeros(T,length(b)))
# 	n = length(b)
# 	rz0 = one(T)
# 	rz1 = one(T)
# 	xtolnb = xtol*norm(b)

# 	k = 0
# 	A_mul_B!(-one(T),A,x,one(T),b)
# 	while norm(b) > xtolnb && k < maxiter
# 		solve!(z,M,b)
# 		k += 1
# 		if k == 1
# 			Ac_mul_B!(one(T),A,z,zero(T),p)
# 			rz1 = one(T)
# 		else
# 			rz1 = dot(b,z)
# 			β = rz1/rz0
# 			Ac_mul_B!(one(T),A,z,β,p)
# 		end
# 		α = rz1/dot(p,p)
# 		A_mul_B!(one(T),A,p,zero(T),Ap)
# 		for i = 1:n
# 			x[i] += α*p[i]
# 			b[i] -= α*Ap[i]
# 		end
# 		rz0 = rz1
# 	end
# 	return x, k
# end

# function richardson(A::TriangularToeplitz, b::Vector, x0::Vector, omega::Float64, maxiter::Int, xtol::Float64)
# 	x = copy(x0)
# 	for i = 1:maxiter
# 		r = b - A*x
# 		if norm(r) < xtol break end
# 		x += omega*r
# 	end
# 	return x
# end
end