# tensoradd.jl
#
# Method for adding one tensor to another according to the
# specified labels, thereby possibly having to permute the
# data. Copying tensors as special case.

# Simple method
# --------------
function tensoradd(A::StridedArray, labelsA, B::StridedArray, labelsB, outputlabels=labelsA)
    dims=size(A)
    T=promote_type(eltype(A), eltype(B))
    C=similar(A, T, dims[indexin(outputlabels, labelsA)])
    tensorcopy!(A, labelsA, C, outputlabels)
    tensoradd!(1, B, labelsB, 1, C, outputlabels)
end
function tensorcopy(A::StridedArray, labelsA, outputlabels=labelsA)
    dims=size(A)
    C=similar(A, dims[indexin(outputlabels, labelsA)])
    tensorcopy!(A, labelsA, C, outputlabels)
end

# In-place method
#-----------------
tensorcopy!(A::StridedArray, labelsA, C::StridedArray, labelsC) = tensoradd!(1, A, labelsA, 0, C, labelsC)

function tensoradd!(alpha::Number, A::StridedArray, labelsA, beta::Number, C::StridedArray, labelsC)
    NA=ndims(A)
    perm=indexin(labelsC, labelsA)
    length(perm) == NA || throw(LabelError("invalid label specification"))
    isperm(perm) || throw(LabelError("labels do not specify a valid permutation"))
    for i = 1:NA
        size(A, perm[i]) == size(C, i) || throw(DimensionMismatch("destination tensor of incorrect size"))
    end
    NA==0 && (C[1]=beta*C[1]+alpha*A[1]; return C)
    perm==collect(1:NA) && return (beta==0 ? scale!(copy!(C, A), alpha) : LinAlg.axpy!(alpha, A, scale!(C, beta)))

    startA, Alinear = _arrayoffset(A)
    startC, Clinear = _arrayoffset(C)

    dims, stridesA, stridesC, minstrides = _addstrides(size(C), _permute(_strides(A), perm), _strides(C))

    if alpha == 0
        beta == 1 || scale!(C,beta)
    elseif alpha == 1 && beta == 0
        tensoradd_rec!(_one, Alinear, _zero, Clinear, dims, startA, startC, stridesA, stridesC, minstrides)
    elseif alpha == 1 && beta == 1
        tensoradd_rec!(_one, Alinear, _one, Clinear, dims, startA, startC, stridesA, stridesC, minstrides)
    elseif beta == 0
        tensoradd_rec!(alpha, Alinear, _zero, Clinear, dims, startA, startC, stridesA, stridesC, minstrides)
    elseif beta == 1
        tensoradd_rec!(alpha, Alinear, _one, Clinear, dims, startA, startC, stridesA, stridesC, minstrides)
    else
        tensoradd_rec!(alpha, Alinear, beta, Clinear, dims, startA, startC, stridesA, stridesC, minstrides)
    end
    return C
end

# Recursive implementation
#--------------------------
@generated function tensoradd_rec!{N}(alpha, A::Array, beta, C::Array, dims::NTuple{N, Int}, startA::Int, startC::Int, stridesA::NTuple{N, Int}, stridesC::NTuple{N, Int}, minstrides::NTuple{N, Int})
    quote
        if 2*prod(dims) <= BASELENGTH
            tensoradd_micro!(alpha, A, beta, C, dims, startA, startC, stridesA, stridesC)
        else
            dmax = _indmax(_memjumps(dims, minstrides))
            @dividebody $N dmax dims startA stridesA startC stridesC begin
                tensoradd_rec!(alpha, A, beta, C, dims, startA, startC, stridesA, stridesC, minstrides)
            end
        end
        return C
    end
end

# Micro kernel at end of recursion
#----------------------------------
@generated function tensoradd_micro!{N}(alpha, A::Array, beta, C::Array, dims::NTuple{N, Integer}, startA::Integer, startC::Integer, stridesA::NTuple{N, Integer}, stridesC::NTuple{N, Integer})
    quote
        @stridedloops($N, dims, indA, startA, stridesA, indC, startC, stridesC, @inbounds C[indC]=axpby(beta,C[indC],alpha,A[indA]))
        return C
    end
end

# Stride calculation
#--------------------
@generated function _addstrides{N}(dims::NTuple{N,Int}, stridesA::NTuple{N,Int}, stridesC::NTuple{N,Int})
    meta = Expr(:meta,:inline)
    minstridesex = Expr(:tuple,[:(min(stridesA[$d],stridesC[$d])) for d = 1:N]...)
    quote
        $meta
        minstrides = $minstridesex
        p = sortperm(collect(minstrides))
        dims = _permute(dims, p)
        stridesA = _permute(stridesA, p)
        stridesC = _permute(stridesC, p)
        minstrides = _permute(minstrides, p)

        return dims, stridesA, stridesC, minstrides
    end
end
