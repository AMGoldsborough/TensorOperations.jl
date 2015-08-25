# tensorcontract.jl
#
# Method for contracting two tensors and adding the result
# to a third tensor, according to the specified labels.
# Method for computing the tensorproduct of two tensors as special
# case.

# Simple method
#---------------
function tensorcontract(A, labelsA, B, labelsB, outputlabels = symdiff(labelsA,labelsB); method::Symbol = :BLAS)
    T = promote_type(eltype(A),eltype(B))
    C = similar_from_indices(T, outputlabels, vcat(labelsA,labelsB), A, B)
    tensorcontract!(1,A,labelsA,'N',B,labelsB,'N',0,C,outputlabels;method = method)
end

function tensorproduct(A, labelsA, B, labelsB, outputlabels = vcat(labelsA,labelsB))
    T = promote_type(eltype(A),eltype(B))
    C = similar_from_indices(T, outputlabels, vcat(labelsA,labelsB), A, B)
    tensorproduct!(1,A,labelsA,B,labelsB,0,C,outputlabels)
end


# In-place method
#-----------------
function tensorcontract!(alpha,A,labelsA,conjA,B,labelsB,conjB,beta,C,labelsC;method::Symbol = :BLAS)
    # Updates C as beta*C+alpha*contract(A,B), whereby the contraction pattern
    # is specified by labelsA, labelsB and labelsC. The iterables labelsA(B,C)
    # should contain a unique label for every index of array A(B,C), such that
    # common labels of A and B correspond to indices that will be contracted.
    # Common labels between A and C or B and C indicate the position of the
    # uncontracted indices of A and B with respect to the indices of C, such
    # that the output array of the contraction can be added to C. Every label
    # should thus appear exactly twice in the union of labelsA, labelsB and
    # labelsC and the associated indices of the tensors should have identical
    # size.
    # Array A and/or B can be also conjugated by setting conjA and/or conjB
    # equal  to 'C' instead of 'N'.
    # The parametric argument method can be specified to choose between two
    # different contraction strategies:
    # -> method = :BLAS : permutes tensors (requires extra memory) and then
    #                   calls built-in (typically BLAS) multiplication
    # -> method = :native : memory-free native julia tensor contraction

    # Get properties of input arrays
    NA = ndims(A)
    NB = ndims(B)
    NC = ndims(C)

    # Process labels, do some error checking and analyse problem structure
    NA == length(labelsA) || throw(LabelError("invalid label length: $labelsA"))
    NB == length(labelsB) || throw(LabelError("invalid label length: $labelsB"))
    NC == length(labelsC) || throw(LabelError("invalid label length: $labelsC"))

    NA == length(unique(labelsA)) || throw(LabelError("handle inner contraction first with tensortrace: $labelsA"))
    NB == length(unique(labelsB)) || throw(LabelError("handle inner contraction first with tensortrace: $labelsB"))
    NC == length(unique(labelsC)) || throw(LabelError("handle inner contraction first with tensortrace: $labelsC"))

    clabels = intersect(labelsA,labelsB)
    numcontract = length(clabels)
    olabelsA = intersect(labelsC,labelsA)
    numopenA = length(olabelsA)
    olabelsB = intersect(labelsC,labelsB)
    numopenB = length(olabelsB)

    if numcontract+numopenA != NA || numcontract+numopenB != NB || numopenA+numopenB != NC
        throw(LabelError("invalid contraction pattern"))
    end

    # Compute contraction indices and check for valid permutation
    cindA = indexin(clabels,labelsA)
    oindA = indexin(olabelsA,labelsA)
    cindB = indexin(clabels,labelsB)
    oindB = indexin(olabelsB,labelsB)
    indCinAB = indexin(labelsC,vcat(olabelsA,olabelsB))

    isperm(vcat(oindA,cindA)) || throw(LabelError("invalid contraction pattern"))
    isperm(vcat(oindB,cindB)) || throw(LabelError("invalid contraction pattern"))
    isperm(indCinAB) || throw(LabelError("invalid contraction pattern"))

    if method == :BLAS
        contract_blas!(alpha,A,conjA,B,conjB,beta,C,oindA,cindA,oindB,cindB,indCinAB)
    elseif method == :native
        contract_native!(alpha,A,conjA,B,conjB,beta,C,oindA,cindA,oindB,cindB,indCinAB)
    else
        throw(ArgumentError("unknown contraction method"))
    end
    return C
end

function tensorproduct!(alpha,A,labelsA,B,labelsB,beta,C,labelsC)
    # Get properties of input arrays
    ndims(A) + ndims(B) == ndims(C) || throw(LabelError("not a valid tensor product"))

    return tensorcontract!(alpha,A,labelsA,'N',B,labelsB,'N',beta,C,labelsC;method = :native)
end

# Implementation methods
#------------------------
# High level: can be extended for other types of arrays or tensors
function contract_blas!(alpha,A::StridedArray,conjA,B::StridedArray,conjB,beta,C::StridedArray,oindA,cindA,oindB,cindB,indCinAB)
    # The :BLAS method specification permutes A and B such that indopen and
    # indcontract are grouped, reshape them to matrices with all indopen on one
    # side and all indcontract on the other. Compute the data for C from
    # multiplying these matrices. Permute again to bring indices in requested
    # order.

    conjA == 'N' || conjA == 'C' || throw(ArgumentError("Value of conjA should be 'N' or 'C' instead of $conjA"))
    conjB == 'N' || conjB == 'C' || throw(ArgumentError("Value of conjB should be 'N' or 'C' instead of $conjB"))

    NA = ndims(A)
    NB = ndims(B)
    NC = ndims(C)
    TA = eltype(A)
    TB = eltype(B)
    TC = eltype(C)

    # dimension checking
    dimA = size(A)
    dimB = size(B)
    dimC = size(C)

    cdimsA = dimA[cindA]
    cdimsB = dimB[cindB]
    odimsA = dimA[oindA]
    odimsB = dimB[oindB]
    odimsAB = tuple(odimsA...,odimsB...)

    for i = 1:length(cdimsA)
        cdimsA[i] == cdimsB[i] || throw(DimensionMismatch())
    end
    cdims = cdimsA

    for i = 1:length(indCinAB)
        dimC[i] == odimsAB[indCinAB[i]] || throw(DimensionMismatch())
    end

    olengthA = prod(odimsA)
    olengthB = prod(odimsB)
    clength = prod(cdims)

    # permute A
    if conjA == 'C'
        pA = vcat(cindA,oindA)
        if pA == collect(1:NA) && TA == TC && isa(A,Array)
            Amat = reshape(A,(clength,olengthA))
        else
            Apermuted = Array{TC}(tuple(cdims...,odimsA...))
            # tensorcopy!(A,1:NA,Apermuted,pA)
            add_native!(1,A,0,Apermuted,pA)
            Amat = reshape(Apermuted,(clength,olengthA))
        end
    else
        if vcat(oindA,cindA) == collect(1:NA) && TA == TC && isa(A,Array)
            Amat = reshape(A,(olengthA,clength))
        elseif vcat(cindA,oindA) == collect(1:NA) && TA == TC && isa(A,Array)
            conjA = 'T'
            Amat = reshape(A,(clength,olengthA))
        else
            pA = vcat(cindA,oindA)
            conjA = 'T' # it is more efficient to compute At*B
            Apermuted = Array{TC}(tuple(cdims...,odimsA...))
            # tensorcopy!(A,1:NA,Apermuted,pA)
            add_native!(1,A,0,Apermuted,pA)
            Amat = reshape(Apermuted,(clength,olengthA))
        end
    end

    # permute B
    if conjB == 'C'
        pB = vcat(oindB,cindB)
        if pB == collect(1:NB) && TB == TC && isa(B,Array)
            Bmat = reshape(B,(olengthB,clength))
        else
            Bpermuted = Array{TC}(tuple(odimsB...,cdims...))
            # tensorcopy!(B,1:NB,Bpermuted,pB)
            add_native!(1,B,0,Bpermuted,pB)
            Bmat = reshape(Bpermuted,(olengthB,clength))
        end
    else
        if vcat(cindB,oindB) == collect(1:NB) && TB == TC && isa(B,Array)
            Bmat = reshape(B,(clength,olengthB))
        elseif vcat(oindB,cindB) == collect(1:NB) && TB == TC && isa(B,Array)
            conjB = 'T'
            Bmat = reshape(B,(olengthB,clength))
        else
            pB = vcat(cindB,oindB)
            Bpermuted = Array{TC}(tuple(cdims...,odimsB...))
            # tensorcopy!(B,1:NB,Bpermuted,pB)
            add_native!(1,B,0,Bpermuted,pB)
            Bmat = reshape(Bpermuted,(clength,olengthB))
        end
    end

    # calculate C
    if indCinAB == collect(1:NC) && isa(C,Array)
        Cmat = reshape(C,(olengthA,olengthB))
        BLAS.gemm!(conjA,conjB,TC(alpha),Amat,Bmat,TC(beta),Cmat)
    else
        Cmat = Array{TC}(olengthA,olengthB)
        BLAS.gemm!(conjA,conjB,TC(1),Amat,Bmat,TC(0),Cmat)
        # tensoradd!(alpha,reshape(Cmat,tuple(odimsA...,odimsB...)),pC,beta,C,1:NC)
        add_native!(alpha,reshape(Cmat,tuple(odimsA...,odimsB...)),beta,C,indCinAB)
    end
    return C
end

# High level: can be extended for other types of arrays or tensors
function contract_native!(alpha,A::StridedArray,conjA::Char,B::StridedArray,conjB::Char,beta,C::StridedArray,oindA,cindA,oindB,cindB,indCinAB)
    # native contraction method using divide and conquer

    conjA == 'N' || conjA == 'C' || throw(ArgumentError("Value of conjA should be 'N' or 'C' instead of $conjA"))
    conjB == 'N' || conjB == 'C' || throw(ArgumentError("Value of conjB should be 'N' or 'C' instead of $conjB"))

    NA = ndims(A)
    NB = ndims(B)
    NC = ndims(C)

    # dimension checking
    dimA = size(A)
    dimB = size(B)
    dimC = size(C)

    cdimsA = dimA[cindA]
    cdimsB = dimB[cindB]
    odimsA = dimA[oindA]
    odimsB = dimB[oindB]
    odimsAB = tuple(odimsA...,odimsB...)

    # Perform contraction
    pA = vcat(oindA,cindA)
    pB = vcat(oindB,cindB)
    sA = _permute(_strides(A),pA)
    sB = _permute(_strides(B),pB)
    sC = _permute(_strides(C),invperm(indCinAB))

    dimsA = _permute(size(A),pA)
    dimsB = _permute(size(B),pB)

    dims, stridesA, stridesB, stridesC, minstrides = _contractstrides(dimsA, dimsB, sA, sB, sC)
    offsetA = offsetB = offsetC = 0
    if conjA == 'N'
        dataA = StridedData(A,stridesA)
    else
        dataA = conj(StridedData(A,stridesA))
    end
    if conjB == 'N'
        dataB = StridedData(B,stridesB)
    else
        dataB = conj(StridedData(B,stridesB))
    end
    dataC = StridedData(C,stridesC)

    # contract via recursive divide and conquer
    if alpha == 0
        beta == 1 || _scale!(dataC,beta,dims)
    elseif alpha == 1 && beta == 0
        contract_rec!(_one,dataA,dataB,_zero,dataC,dims,offsetA,offsetB,offsetC,minstrides)
    elseif alpha == 1 && beta == 1
        contract_rec!(_one,dataA,dataB,_one,dataC,dims,offsetA,offsetB,offsetC,minstrides)
    elseif beta == 0
        contract_rec!(alpha,dataA,dataB,_zero,dataC,dims,offsetA,offsetB,offsetC,minstrides)
    elseif beta == 1
        contract_rec!(alpha,dataA,dataB,_one,dataC,dims,offsetA,offsetB,offsetC,minstrides)
    else
        contract_rec!(alpha,dataA,dataB,beta,dataC,dims,offsetA,offsetB,offsetC,minstrides)
    end
    return C
end

# Recursive divide and conquer approach:
@generated function contract_rec!{N}(alpha, A::StridedData{N}, B::StridedData{N}, beta, C::StridedData{N},
    dims::NTuple{N, Int}, offsetA::Int, offsetB::Int, offsetC::Int, minstrides::NTuple{N, Int})

    quote
        odimsA = _filterdims(_filterdims(dims,A),C)
        odimsB = _filterdims(_filterdims(dims,B),C)
        cdims = _filterdims(_filterdims(dims,A),B)
        oAlength = prod(odimsA)
        oBlength = prod(odimsB)
        clength = prod(cdims)

        if oAlength*oBlength + clength*(oAlength+oBlength) <= BASELENGTH
            contract_micro!(alpha,A,B,beta,C,dims,offsetA,offsetB,offsetC)
        else
            if clength > oAlength && clength > oBlength
                dmax = _indmax(_memjumps(cdims, minstrides))
            elseif oAlength > oBlength
                dmax = _indmax(_memjumps(odimsA, minstrides))
            else
                dmax = _indmax(_memjumps(odimsB, minstrides))
            end
            @dividebody $N dmax dims offsetA A offsetB B offsetC C begin
                    contract_rec!(alpha,A,B,beta,C,dims,offsetA,offsetB,offsetC,minstrides)
                end begin
                if C.strides[dmax] == 0 # dmax is contraction dimension: beta -> 1
                    contract_rec!(alpha,A,B,_one,C,dims,offsetA,offsetB,offsetC,minstrides)
                else
                    contract_rec!(alpha,A,B,beta,C,dims,offsetA,offsetB,offsetC,minstrides)
                end
            end
        end
        return C
    end
end

# Micro kernel at end of recursion
@generated function contract_micro!{N}(alpha, A::StridedData{N}, B::StridedData{N}, beta, C::StridedData{N}, dims::NTuple{N, Int}, offsetA, offsetB, offsetC)
    quote
        _scale!(C, beta, dims, offsetC)
        startA = A.start+offsetA
        stridesA = A.strides
        startB = B.start+offsetB
        stridesB = B.strides
        startC = C.start+offsetC
        stridesC = C.strides

        @stridedloops($N, dims, indA, startA, stridesA, indB, startB, stridesB, indC, startC, stridesC, @inbounds C[indC] = axpby(alpha,A[indA]*B[indB],_one,C[indC]))
        return C
    end
end

# Stride calculation
#--------------------
@generated function _contractstrides{NA,NB,NC}(dimsA::NTuple{NA,Int}, dimsB::NTuple{NB,Int},
    stridesA::NTuple{NA,Int}, stridesB::NTuple{NB,Int}, stridesC::NTuple{NC,Int})
    meta = Expr(:meta,:inline)
    cN = div(NA+NB-NC,2)
    oNA = NA - cN
    oNB = NB - cN

    dimsex = Expr(:tuple,[:(dimsA[$d]) for d = 1:oNA]...,[:(dimsB[$d]) for d = 1:oNB]...,[:(dimsA[$(oNA+d)]) for d = 1:cN]...)

    stridesAex = Expr(:tuple,[:(stridesA[$d]) for d = 1:oNA]...,[0 for d = 1:oNB]...,[:(stridesA[$(oNA+d)]) for d = 1:cN]...)
    stridesBex = Expr(:tuple,[0 for d = 1:oNA]...,[:(stridesB[$d]) for d = 1:oNB]...,[:(stridesB[$(oNB+d)]) for d = 1:cN]...)
    stridesCex = Expr(:tuple,[:(stridesC[$d]) for d = 1:(oNA+oNB)]...,[0 for d = 1:cN]...)

    minstridesex = Expr(:tuple,[:(min(stridesA[$d],stridesC[$d])) for d = 1:oNA]...,
    [:(min(stridesB[$d],stridesC[$(oNA+d)])) for d = 1:oNB]...,
    [:(min(stridesA[$(oNA+d)],stridesB[$(oNB+d)])) for d = 1:cN]...)
    quote
        $meta
        minstrides = $minstridesex
        p = sortperm(collect(minstrides))
        dims = _permute($dimsex, p)
        stridesA = _permute($stridesAex, p)
        stridesB = _permute($stridesBex, p)
        stridesC = _permute($stridesCex, p)
        minstrides = _permute(minstrides, p)

        return dims, stridesA, stridesB, stridesC, minstrides
    end
end
