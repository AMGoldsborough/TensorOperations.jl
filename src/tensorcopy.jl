# tensorcopy.jl
#
# Method for copying one tensor to another according to the
# specified labels, thereby possibly having to permute the
# data.

# Simple method
# --------------
function tensorcopy(A::StridedArray, labelsA, outputlabels=labelsA)
    dims=size(A)
    C=similar(A, dims[indexin(outputlabels, labelsA)])
    tensorcopy!(A, labelsA, C, outputlabels)
end

# In-place method
#-----------------
function tensorcopy!(A::StridedArray, labelsA, C::StridedArray, labelsC)
    N = ndims(A)
    ndims(C) == length(labelsC) == length(labelsA) == N || throw(LabelError("invalid label specification"))

    perm=indexin(labelsC, labelsA)
    isperm(perm) || throw(LabelError("labels do not specify a valid permutation"))
    for i = 1:N
        size(A, perm[i]) == size(C, i) || throw(DimensionMismatch("destination tensor of incorrect size"))
    end
    perm==collect(1:N) && return copy!(C, A)

    startA, Alinear = _arrayoffset(A)
    startC, Clinear = _arrayoffset(C)

    stridesA = _permute(_strides(A), perm)
    stridesC = _strides(C)
    minstrides = _min(stridesA, stridesC)
    p = sortperm(collect(minstrides))

    dims = _permute(size(C), p)
    stridesA = _permute(stridesA, p)
    stridesC = _permute(stridesC, p)
    minstrides = _permute(minstrides, p)

    tensoradd_rec!(_one,Alinear, _zero, Clinear, dims, startA, startC, stridesA, stridesC, minstrides)
    return C
end
