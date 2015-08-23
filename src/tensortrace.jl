# tensortrace.jl
#
# Method for tracing some of the indices of a tensor and
# adding the result to another tensor.

# Simple method
#---------------
function tensortrace(A::StridedArray,labelsA,outputlabels)
    dimsA=size(A)
    C=similar(A,dimsA[indexin(outputlabels,labelsA)])
    tensortrace!(1,A,labelsA,0,C,outputlabels)
end
function tensortrace(A::StridedArray,labelsA) # there is no one-line method to compute the default outputlabels
    ulabelsA=unique(labelsA)
    labelsC=similar(labelsA,0)
    sizehint!(labelsC,length(ulabelsA))
    for j=1:length(ulabelsA)
        ind=findfirst(labelsA,ulabelsA[j])
        if findnext(labelsA,ulabelsA[j],ind+1)==0
            push!(labelsC,ulabelsA[j])
        end
    end
    tensortrace(A,labelsA,labelsC)
end

# In-place method
#-----------------
function tensortrace!(alpha::Number,A::StridedArray,labelsA,beta::Number,C::StridedArray,labelsC)
    NA=ndims(A)
    NC=ndims(C)
    (length(labelsA)==NA && length(labelsC)==NC) || throw(LabelError("invalid label specification"))
    NA==NC && return tensoradd!(alpha,A,labelsA,beta,C,labelsC) # nothing to trace

    oindA=indexin(labelsC,labelsA)
    clabels=unique(setdiff(labelsA,labelsC))
    NA==NC+2*length(clabels) || throw(LabelError("invalid label specification"))

    cindA1=Array(Int,length(clabels))
    cindA2=Array(Int,length(clabels))
    for i=1:length(clabels)
        cindA1[i]=findfirst(labelsA,clabels[i])
        cindA2[i]=findnext(labelsA,clabels[i],cindA1[i]+1)
    end
    pA = vcat(oindA, cindA1, cindA2)
    isperm(pA) || throw(LabelError("invalid label specification"))

    for i = 1:NC
        size(A,oindA[i]) == size(C,i) || throw(DimensionMismatch("tensor sizes incompatible"))
    end
    for i = 1:div(NA-NC,2)
        size(A,cindA1[i]) == size(A,cindA2[i]) || throw(DimensionMismatch("tensor sizes incompatible"))
    end

    dims, stridesA, stridesC, minstrides = _tracestrides(_permute(size(A),pA), _permute(_strides(A),pA), _strides(C))
    dataA = StridedData(A,stridesA)
    offsetA = 0
    dataC = StridedData(C,stridesC)
    offsetC = 0

    if alpha == 0
        beta == 1 || scale!(C,beta)
    elseif alpha == 1 && beta == 0
        trace_rec!(_one, dataA, _zero, dataC, dims, offsetA, offsetC, minstrides)
    elseif alpha == 1 && beta == 1
        trace_rec!(_one, dataA, _one, dataC, dims, offsetA, offsetC, minstrides)
    elseif beta == 0
        trace_rec!(alpha, dataA, _zero, dataC, dims, offsetA, offsetC, minstrides)
    elseif beta == 1
        trace_rec!(alpha, dataA, _one, dataC, dims, offsetA, offsetC, minstrides)
    else
        trace_rec!(alpha, dataA, beta, dataC, dims, offsetA, offsetC, minstrides)
    end
    return C
end

# Recursive implementation
#--------------------------
@generated function trace_rec!{N}(alpha, A::StridedData{N}, beta, C::StridedData{N}, dims::NTuple{N, Int}, offsetA::Int, offsetC::Int, minstrides::NTuple{N, Int})
    quote
        if prod(dims) + prod(_filterdims(dims,C)) <= 2*BASELENGTH
            trace_micro!(alpha, A, beta, C, dims, offsetA, offsetC)
        else
            dmax = _indmax(_memjumps(dims, minstrides))
            @dividebody $N dmax dims offsetA A offsetC C begin
                trace_rec!(alpha, A, beta, C, dims, offsetA, offsetC, minstrides)
            end begin
                if C.strides[dmax] == 0
                    trace_rec!(alpha, A, _one, C, dims, offsetA, offsetC, minstrides)
                else
                    trace_rec!(alpha, A, beta, C, dims, offsetA, offsetC, minstrides)
                end
            end
        end
        return C
    end
end

# Micro kernel at end of recursion
#----------------------------------
@generated function trace_micro!{N}(alpha, A::StridedData{N}, beta, C::StridedData{N}, dims::NTuple{N, Int}, offsetA::Int, offsetC::Int)
    quote
        _scale!(C, beta, dims, offsetC)
        startA = A.start+offsetA
        stridesA = A.strides
        startC = C.start+offsetC
        stridesC = C.strides
        @stridedloops($N, dims, indA, startA, stridesA, indC, startC, stridesC, @inbounds C[indC]=axpby(alpha,A[indA],_one,C[indC]))
        return C
    end
end

# Stride calculation
#--------------------
@generated function _tracestrides{NA,NC}(dims::NTuple{NA,Int}, stridesA::NTuple{NA,Int}, stridesC::NTuple{NC,Int})
    M = div(NA-NC,2)
    dimsex = Expr(:tuple,[:(dims[$d]) for d=1:(NC+M)]...)
    stridesAex = Expr(:tuple,[:(stridesA[$d]) for d = 1:NC]...,[:(stridesA[$(NC+d)]+stridesA[$(NC+M+d)]) for d = 1:M]...)
    stridesCex = Expr(:tuple,[:(stridesC[$d]) for d = 1:NC]...,[0 for d = 1:M]...)
    minstridesex = Expr(:tuple,[:(min(stridesA[$d],stridesC[$d])) for d = 1:NC]...,[:(stridesA[$(NC+d)]+stridesA[$(NC+M+d)]) for d = 1:M]...)
    quote
        minstrides = $minstridesex
        p = sortperm(collect(minstrides))
        newdims = _permute($dimsex, p)
        newstridesA = _permute($stridesAex, p)
        newstridesC = _permute($stridesCex, p)
        minstrides = _permute(minstrides, p)

        return newdims, newstridesA, newstridesC, minstrides
    end
end
