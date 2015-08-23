# tensorcontract.jl
#
# Method for contracting two tensors and adding the result
# to a third tensor, according to the specified labels.
# Method for computing the tensorproduct of two tensors as special
# case.

# Simple method
#---------------
function tensorcontract(A::StridedArray,labelsA,B::StridedArray,labelsB,outputlabels = symdiff(labelsA,labelsB);method::Symbol = :BLAS)
    dimsA = size(A)
    dimsB = size(B)
    dimsC = tuple(dimsA...,dimsB...)
    dimsC = dimsC[indexin(outputlabels,vcat(labelsA,labelsB))]
    T = promote_type(eltype(A),eltype(B))
    C = similar(A,T,dimsC)
    tensorcontract!(1,A,labelsA,'N',B,labelsB,'N',0,C,outputlabels;method = method)
end

function tensorproduct(A::StridedArray,labelsA,B::StridedArray,labelsB,outputlabels = vcat(labelsA,labelsB))
    dimsA = size(A)
    dimsB = size(B)
    dimsC = tuple(dimsA...,dimsB...)
    dimsC = dimsC[indexin(outputlabels,vcat(labelsA,labelsB))]
    T = promote_type(eltype(A),eltype(B))
    C = similar(A,T,dimsC)
    tensorproduct!(1,A,labelsA,B,labelsB,0,C,outputlabels)
end


# In-place method
#-----------------
function tensorcontract!(alpha::Number,A::StridedArray,labelsA,conjA::Char,B::StridedArray,labelsB,conjB::Char,beta::Number,C::StridedArray,labelsC;method::Symbol = :BLAS)
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
    if NA != length(labelsA) || NB != length(labelsB) || NC != length(labelsC)
        throw(LabelError("invalid label specification"))
    end
    ulabelsA = unique(labelsA)
    ulabelsB = unique(labelsB)
    ulabelsC = unique(labelsC)
    if NA != length(ulabelsA) || NB != length(ulabelsB) || NC != length(ulabelsC)
        throw(LabelError("tensorcontract requires unique label for every index of the tensor, handle inner contraction first with tensortrace"))
    end

    clabels = intersect(ulabelsA,ulabelsB)
    numcontract = length(clabels)
    olabelsA = intersect(ulabelsC,ulabelsA)
    numopenA = length(olabelsA)
    olabelsB = intersect(ulabelsC,ulabelsB)
    numopenB = length(olabelsB)

    if numcontract+numopenA != NA || numcontract+numopenB != NB || numopenA+numopenB != NC
        throw(LabelError("invalid contraction pattern"))
    end

    conjA == 'N' || conjA == 'C' || throw(ArgumentError("Value of conjA should be 'N' or 'C'"))
    conjB == 'N' || conjB == 'C' || throw(ArgumentError("Value of conjB should be 'N' or 'C'"))

    # Compute and contraction indices and check size compatibility
    cindA = indexin(clabels,ulabelsA)
    oindA = indexin(olabelsA,ulabelsA)
    oindCA = indexin(olabelsA,ulabelsC)
    cindB = indexin(clabels,ulabelsB)
    oindB = indexin(olabelsB,ulabelsB)
    oindCB = indexin(olabelsB,ulabelsC)

    dimA = size(A)
    dimB = size(B)
    dimC = size(C)

    cdimsA = dimA[cindA]
    cdimsB = dimB[cindB]
    odimsA = dimA[oindA]
    odimsB = dimB[oindB]

    for i = 1:numcontract
        cdimsA[i] == cdimsB[i] || throw(DimensionMismatch("dimension mismatch for label $(clabels[i])"))
    end
    for i = 1:numopenA
        odimsA[i] == dimC[oindCA[i]] || throw(DimensionMismatch("dimension mismatch for label $(olabelsA[i])"))
    end
    for i = 1:numopenB
        odimsB[i] == dimC[oindCB[i]] || throw(DimensionMismatch("dimension mismatch for label $(olabelsB[i])"))
    end

    # Perform contraction
    if method == :BLAS
        contract_blas!(alpha,A,conjA,B,conjB,beta,C,oindA,cindA,oindB,cindB,oindCA,oindCB)
    elseif method == :native
        contract_native!(alpha,A,conjA,B,conjB,beta,C,oindA,cindA,oindB,cindB,oindCA,oindCB)
    else
        throw(ArgumentError("unknown contraction method"))
    end
    return C
end

function tensorproduct!(alpha::Number,A::StridedArray,labelsA,B::StridedArray,labelsB,beta::Number,C::StridedArray,labelsC)
    # Updates C as beta*C+alpha*tensorproduct(A,B), whereby the order of indices
    # in A, B and C are specified by the labels.

    # Get properties of input arrays
    NA = ndims(A)
    NB = ndims(B)
    NC = ndims(C)

    # Process labels, do some error checking and analyse problem structure
    if NA != length(labelsA) || NB != length(labelsB) || NC != length(labelsC)
        throw(LabelError("invalid label specification"))
    end
    NC == NA+NB || throw(LabelError("not a valid tensor product specification"))
    labels = setdiff(labelsC,labelsA)
    labels = setdiff(labels,labelsB)

    isempty(labels) || throw(LabelError("not a valid tensor product specification"))

    return tensorcontract!(alpha,A,labelsA,'N',B,labelsB,'N',beta,C,labelsC;method = :native)
end

# Implementations
#-----------------
function contract_blas!(alpha,A::StridedArray,conjA::Char,B::StridedArray,conjB::Char,beta,C::StridedArray,oindA,cindA,oindB,cindB,oindCA,oindCB)
    # The :BLAS method specification permutes A and B such that indopen and
    # indcontract are grouped, reshape them to matrices with all indopen on one
    # side and all indcontract on the other. Compute the data for C from
    # multiplying these matrices. Permute again to bring indices in requested
    # order.

    NA = ndims(A)
    NB = ndims(B)
    NC = ndims(C)
    TA = eltype(A)
    TB = eltype(B)
    TC = eltype(C)

    # only basic checking, this function is not expected to be called directly
    length(oindA) == length(oindCA) || throw(DimensionMismatch("invalid contraction pattern"))
    length(oindB) == length(oindCB) || throw(DimensionMismatch("invalid contraction pattern"))
    length(cindA) == length(cindB) == div(NA+NB-NC,2) || throw(DimensionMismatch("invalid contraction pattern"))
    length(oindA)+length(cindA) == NA || throw(DimensionMismatch("invalid contraction pattern"))
    length(oindB)+length(cindB) == NB || throw(DimensionMismatch("invalid contraction pattern"))
    length(oindCA)+length(oindCB) == NC || throw(DimensionMismatch("invalid contraction pattern"))

    # try to avoid extra allocation as much as possible
    if NC>0 && vcat(oindCB,oindCA) == collect(1:NC) # better to change role of A and B
        oindA,oindB = oindB,oindA
        cindA,cindB = cindB,cindA
        oindCA,oindCB = oindCB,oindCA
        A,B = B,A
        NA,NB = NB,NA
        TA,TB = TB,TA
    end

    dimsA = size(A)
    odimsA = dimsA[oindA]
    dimsB = size(B)
    odimsB = dimsB[oindB]
    cdims = dimsA[cindA]

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
            tensorcopy!(A,1:NA,Apermuted,pA)
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
            tensorcopy!(A,1:NA,Apermuted,pA)
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
            tensorcopy!(B,1:NB,Bpermuted,pB)
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
            tensorcopy!(B,1:NB,Bpermuted,pB)
            Bmat = reshape(Bpermuted,(clength,olengthB))
        end
    end

    # calculate C
    pC = vcat(oindCA,oindCB)
    if pC == collect(1:NC) && isa(C,Array)
        Cmat = reshape(C,(olengthA,olengthB))
        BLAS.gemm!(conjA,conjB,TC(alpha),Amat,Bmat,TC(beta),Cmat)
    else
        Cmat = Array{TC}(olengthA,olengthB)
        BLAS.gemm!(conjA,conjB,TC(1),Amat,Bmat,TC(0),Cmat)
        tensoradd!(alpha,reshape(Cmat,tuple(odimsA...,odimsB...)),pC,beta,C,1:NC)
    end
    return C
end

function contract_native!(alpha,A::StridedArray,conjA::Char,B::StridedArray,conjB::Char,beta,C::StridedArray,oindA,cindA,oindB,cindB,oindCA,oindCB)
    # only basic checking, this function is not expected to be called directly
    NA = ndims(A)
    NB = ndims(B)
    NC = ndims(C)
    length(oindA) == length(oindCA) || throw(DimensionMismatch("invalid contraction pattern"))
    length(oindB) == length(oindCB) || throw(DimensionMismatch("invalid contraction pattern"))
    length(cindA) == length(cindB) == div(NA+NB-NC,2) || throw(DimensionMismatch("invalid contraction pattern"))
    length(oindA) + length(cindA) == NA || throw(DimensionMismatch("invalid contraction pattern"))
    length(oindB) + length(cindB) == NB || throw(DimensionMismatch("invalid contraction pattern"))
    length(oindCA) + length(oindCB) == NC || throw(DimensionMismatch("invalid contraction pattern"))

    sA = _permute(_strides(A),vcat(oindA,cindA))
    sB = _permute(_strides(B),vcat(oindB,cindB))
    sC = _permute(_strides(C),vcat(oindCA,oindCB))

    dimsA = _permute(size(A),vcat(oindA,cindA))
    dimsB = _permute(size(B),vcat(oindB,cindB))

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

    if alpha == 0
        beta == 1 || scale!(C,beta)
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

# Recursive implementation
#--------------------------
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
#----------------------------------
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
