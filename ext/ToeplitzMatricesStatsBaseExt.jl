module ToeplitzMatricesStatsBaseExt

using ToeplitzMatrices
using StatsBase

function StatsBase.levinson(A::AbstractToeplitz, B::AbstractVecOrMat)
	StatsBase.levinson!(zeros(size(B)), A, copy(B))
end

# extend levinson
function StatsBase.levinson!(x::StridedVector, A::SymmetricToeplitz, b::StridedVector)
	StatsBase.levinson!(A.vc, b, x)
end

function StatsBase.levinson!(C::StridedMatrix, A::SymmetricToeplitz, B::StridedMatrix)
    n = size(B, 2)
    if n != size(C, 2)
        throw(DimensionMismatch("input and output matrices must have same number of columns"))
    end
    for j = 1:n
        StatsBase.levinson!(view(C, :, j), A, view(B, :, j))
    end
    C
end

end
