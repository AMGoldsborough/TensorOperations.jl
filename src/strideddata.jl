

immutable StridedData{T,N,C}
    data::Vector{T}
    strides::NTuple{N,Int}
    offset::Int
end

StridedData{T,N}(a::Array{T,N}) = StridedData{T,N,false}(vec(a),_strides(a),0)
StridedData{T,N,A<:Array,I<:Tuple{Vararg{Union(Colon,Range{Int64},Int64)}}}(a::SubArray{T,N,A,I}) =
    StridedData{T,N,false}(vec(a.parent),_strides(a),a.first_index-1)

Base.getindex{T,N}(a::StridedData{T,N,false},i) = a.data[i]
Base.getindex{T,N}(a::StridedData{T,N,true},i) = conj(a.data[i])

Base.setindex!{T,N}(a::StridedData{T,N,false},v,i) = (a.data[i] = v)
Base.setindex!{T,N}(a::StridedData{T,N,true},v,i) = (a.data[i] = conj(v))

Base.conj{T,N}(a::StridedData{T,N,false}) = StridedData{T,N,true}(a.data,a.strides,a.offset)
Base.conj{T,N}(a::StridedData{T,N,true}) = StridedData{T,N,false}(a.data,a.strides,a.offset)

+{T,N,C}(a::StridedData{T,N,C},i::Int) = StridedData{T,N,C}(a.data,a.strides,a.offset+i)
-{T,N,C}(a::StridedData{T,N,C},i::Int) = StridedData{T,N,C}(a.data,a.strides,a.offset-i)

_permutestrides{T,N,C}(a::StridedData{T,N,C},p) = StridedData{T,N,C}(a.data,_permute(a.strides,p),a.offset)
