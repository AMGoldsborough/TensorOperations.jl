# tensoradd.jl
#
# Method for adding one tensor to another according to the
# specified labels, thereby possibly having to permute the
# data. Copying as special case.

# Simple methods
# ---------------
function tensorcopy(A, labelsA, outputlabels=labelsA)
    C = similar_from_indices(eltype(A), outputlabels, labelsA, A)
    tensorcopy!(A, labelsA, C, outputlabels)
end

function tensoradd(A, labelsA, B, labelsB, outputlabels=labelsA)
    T = promote_type(eltype(A), eltype(B))
    C = similar_from_indices(T, outputlabels, labelsA, A)
    tensorcopy!(A, labelsA, C, outputlabels)
    tensoradd!(1, B, labelsB, 1, C, outputlabels)
end

# In-place method
#-----------------
tensorcopy!(A, labelsA, C, labelsC) =
    tensoradd!(1, A, labelsA, 0, C, labelsC)

function tensoradd!(alpha,A,labelsA,beta,C,labelsC)
    NA=ndims(A)
    NC=ndims(C)
    length(labelsA)==NA || throw(LabelError("invalid label length: $labelsA"))
    length(labelsC)==NC || throw(LabelError("invalid label length: $labelsC"))

    indCinA=indexin(labelsC,labelsA)
    isperm(indCinA) || throw(LabelError("non-matching labels: $labelsA vs $labelsC"))

    add_native!(alpha, A, beta, C, indCinA)
    return C
end

# Implementation methods
#------------------------
# High level: can be extended for other types of arrays or tensors
function add_native!(alpha, A::StridedArray, beta, C::StridedArray, indCinA)
    for i = 1:ndims(C)
        size(A,indCinA[i]) == size(C,i) || throw(DimensionMismatch())
    end

    dims, stridesA, stridesC, minstrides = _addstrides(size(C), _permute(_strides(A),indCinA), _strides(C))
    dataA = StridedData(A,stridesA)
    offsetA = 0
    dataC = StridedData(C,stridesC)
    offsetC = 0

    if alpha == 0
        beta == 1 || _scale!(dataC,beta,dims)
    elseif alpha == 1 && beta == 0
        add_rec!(_one, dataA, _zero, dataC, dims, offsetA, offsetC, minstrides)
    elseif alpha == 1 && beta == 1
        add_rec!(_one, dataA, _one, dataC, dims, offsetA, offsetC, minstrides)
    elseif beta == 0
        add_rec!(alpha, dataA, _zero, dataC, dims, offsetA, offsetC, minstrides)
    elseif beta == 1
        add_rec!(alpha, dataA, _one, dataC, dims, offsetA, offsetC, minstrides)
    else
        add_rec!(alpha, dataA, beta, dataC, dims, offsetA, offsetC, minstrides)
    end
    return C
end

# Recursive divide and conquer approach:
@generated function add_rec!{N}(alpha, A::StridedData{N}, beta, C::StridedData{N}, dims::NTuple{N, Int}, offsetA::Int, offsetC::Int, minstrides::NTuple{N, Int})
    quote
        if prod(dims) <= BASELENGTH
            add_micro!(alpha, A, beta, C, dims, offsetA, offsetC)
        else
            dmax = _indmax(_memjumps(dims, minstrides))
            @dividebody $N dmax dims offsetA A offsetC C begin
                add_rec!(alpha, A, beta, C, dims, offsetA, offsetC, minstrides)
            end begin
                add_rec!(alpha, A, beta, C, dims, offsetA, offsetC, minstrides)
            end
        end
        return C
    end
end

# Micro kernel at end of recursion:
@generated function add_micro!{N}(alpha, A::StridedData{N}, beta, C::StridedData{N}, dims::NTuple{N, Int}, offsetA::Int, offsetC::Int)
    quote
        startA = A.start+offsetA
        stridesA = A.strides
        startC = C.start+offsetC
        stridesC = C.strides
        @stridedloops($N, dims, indA, startA, stridesA, indC, startC, stridesC, @inbounds C[indC]=axpby(alpha,A[indA],beta,C[indC]))
        return C
    end
end

# Stride calculation
#--------------------
@generated function _addstrides{N}(dims::NTuple{N,Int}, stridesA::NTuple{N,Int}, stridesC::NTuple{N,Int})
    minstridesex = Expr(:tuple,[:(min(stridesA[$d],stridesC[$d])) for d = 1:N]...)
    quote
        minstrides = $minstridesex
        p = sortperm(collect(minstrides))
        dims = _permute(dims, p)
        stridesA = _permute(stridesA, p)
        stridesC = _permute(stridesC, p)
        minstrides = _permute(minstrides, p)

        return dims, stridesA, stridesC, minstrides
    end
end
