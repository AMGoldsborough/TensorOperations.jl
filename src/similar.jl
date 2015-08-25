function similar_from_indices{T}(::Type{T}, dstlabels, srclabels, A::Vararg{StridedArray})
    dims = collectdims(A...)
    p = indexin(dstlabels, srclabels)
    return Array{T}(dims[p])
end

@inline collectdims(A::StridedArray) = size(A)
@inline collectdims(A::Vararg{StridedArray}) = tuple(size(A[1])..., collectdims(Base.tail(A)...)...)
