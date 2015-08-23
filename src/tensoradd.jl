# tensoradd.jl
#
# Method for adding one tensor to another according to the
# specified labels, thereby possibly having to permute the
# data. Copying as special case.

# Simple methods
# ---------------
function tensorcopy(A::StridedArray, labelsA, outputlabels=labelsA)
    dims=size(A)
    C=similar(A, dims[indexin(outputlabels, labelsA)])
    tensorcopy!(A, labelsA, C, outputlabels)
end

function tensoradd(A::StridedArray, labelsA, B::StridedArray, labelsB, outputlabels=labelsA)
    dims=size(A)
    T=promote_type(eltype(A), eltype(B))
    C=similar(A, T, dims[indexin(outputlabels, labelsA)])
    tensorcopy!(A, labelsA, C, outputlabels)
    tensoradd!(1, B, labelsB, 1, C, outputlabels)
end

# In-place method
#-----------------
tensorcopy!(A::StridedArray, labelsA, C::StridedArray, labelsC) =
    tensoradd!(1, A, labelsA, 0, C, labelsC)

function tensoradd!(alpha::Number,A::StridedArray,labelsA,beta::Number,C::StridedArray,labelsC)
    NA=ndims(A)
    NC=ndims(C)
    (length(labelsA)==NA && length(labelsC)==NC) || throw(LabelError("invalid label specification"))

    pA=indexin(labelsC,labelsA)
    isperm(pA) || throw(LabelError("invalid label specification"))

    for i = 1:NC
        size(A,pA[i]) == size(C,i) || throw(DimensionMismatch("tensor sizes incompatible"))
    end

    dims, stridesA, stridesC, minstrides = _addstrides(size(C), _permute(_strides(A),pA), _strides(C))
    dataA = StridedData(A,stridesA)
    offsetA = 0
    dataC = StridedData(C,stridesC)
    offsetC = 0

    if alpha == 0
        beta == 1 || scale!(C,beta)
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

# Recursive implementation
#--------------------------
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

# Micro kernel at end of recursion
#----------------------------------
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
