# tensorcopy.jl
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

    startA, Alinear = _arrayoffset(A)
    startC, Clinear = _arrayoffset(C)

    dims, stridesA, stridesC, minstrides = _tracestrides(_permute(size(A),pA), _permute(_strides(A),pA), _strides(C))

    if alpha == 0
        beta == 1 || scale!(C,beta)
    elseif alpha == 1 && beta == 0
        tensortrace_rec!(_one, Alinear, _zero, Clinear, dims, startA, startC, stridesA, stridesC, minstrides)
    elseif alpha == 1 && beta == 1
        tensortrace_rec!(_one, Alinear, _one, Clinear, dims, startA, startC, stridesA, stridesC, minstrides)
    elseif beta == 0
        tensortrace_rec!(alpha, Alinear, _zero, Clinear, dims, startA, startC, stridesA, stridesC, minstrides)
    elseif beta == 1
        tensortrace_rec!(alpha, Alinear, _one, Clinear, dims, startA, startC, stridesA, stridesC, minstrides)
    else
        tensortrace_rec!(alpha, Alinear, beta, Clinear, dims, startA, startC, stridesA, stridesC, minstrides)
    end
    return C
end

# Recursive implementation
#--------------------------
@generated function tensortrace_rec!{N}(alpha, A::Array, beta, C::Array, dims::NTuple{N, Int}, startA::Int, startC::Int, stridesA::NTuple{N, Int}, stridesC::NTuple{N, Int}, minstrides::NTuple{N, Int})
    quote
        @show dims, startA, startC, stridesA, stridesC
        @show alpha, beta
        @show _size(dims,stridesA)+_size(dims,stridesC) 
        if _size(dims,stridesA)+_size(dims,stridesC) <= 2*BASELENGTH
            tensoradd_micro!(alpha, A, beta, C, dims, startA, startC, stridesA, stridesC)
        else
            dmax = _indmax(_memjumps(dims, minstrides))
            if stridesC[dmax] == 0
                @dividebody2 $N dmax dims startA stridesA startC stridesC begin
                    tensortrace_rec!(alpha, A, beta, C, dims, startA, startC, stridesA, stridesC, minstrides)
                end begin
                    tensortrace_rec!(alpha, A, _one, C, dims, startA, startC, stridesA, stridesC, minstrides)
                end
            else
                @dividebody $N dmax dims startA stridesA startC stridesC begin
                    tensortrace_rec!(alpha, A, beta, C, dims, startA, startC, stridesA, stridesC, minstrides)
                end
            end
        end
        return C
    end
end

# Stride calculation
#--------------------
@generated function _tracestrides{NA,NC}(dims::NTuple{NA,Int}, stridesA::NTuple{NA,Int}, stridesC::NTuple{NC,Int})
    M = div(NA-NC,2)
    meta = Expr(:meta,:inline)
    dimsex = Expr(:tuple,[:(dims[$d]) for d=1:(NC+M)]...)
    stridesAex = Expr(:tuple,[:(stridesA[$d]) for d = 1:NC]...,[:(stridesA[$(NC+d)]+stridesA[$(NC+M+d)]) for d = 1:M]...)
    stridesCex = Expr(:tuple,[:(stridesC[$d]) for d = 1:NC]...,[0 for d = 1:M]...)
    minstridesex = Expr(:tuple,[:(min(stridesA[$d],stridesC[$d])) for d = 1:NC]...,[:(stridesA[$(NC+d)]+stridesA[$(NC+M+d)]) for d = 1:M]...)
    quote
        $meta
        minstrides = $minstridesex
        p = sortperm(collect(minstrides))
        newdims = _permute($dimsex, p)
        newstridesA = _permute($stridesAex, p)
        newstridesC = _permute($stridesCex, p)
        minstrides = _permute(minstrides, p)

        return newdims, newstridesA, newstridesC, minstrides
    end
end
