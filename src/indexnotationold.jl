macro tensor(arg)
    esc(tensorify(arg))
end

function tensorify(ex::Expr)
    ex.head==:block && return Expr(:block,map(tensorify,ex.args))
    
    e





#
# # LabeledArray
# #--------------
# # Wraps an Array with a LabelList. This type acts as return type
# # of getindex(::Array,::LabelList) and can engage in tensor operations.
# type LabeledArray{T,N}
#     data::StridedArray{T,N}
#     labels::Vector{Symbol}
#     function LabeledArray(data::StridedArray{T,N},labels::Vector{Symbol})
#         if length(labels)!=N || length(unique(labels))!=N
#             throw(LabelError("Provide one unique label per index"))
#         end
#         new(data,labels)
#     end
# end
# function LabeledArray{T}(data::StridedArray{T},labels::Vector{Symbol})
#     if length(labels)!=ndims(data)
#         throw(LabelError("Provide one label per index"))
#     end
#     ulabels=unique(labels)
#     if length(ulabels)!=ndims(data) # there are some internal traces
#         newlabels=similar(ulabels,0)
#         for j=1:length(ulabels)
#             ind=findfirst(labels,ulabels[j])
#             ind2=findnext(labels,ulabels[j],ind+1)
#             if ind2==0
#                 push!(newlabels,ulabels[j])
#             elseif findnext(labels,ulabels[j],ind2+1)!=0
#                 throw(LabelError("Label can appear at most twice in a single array"))
#             end
#         end
#         newdata=tensortrace(data,labels,newlabels)
#     else
#         newdata=data
#         newlabels=labels
#     end
#     N=ndims(newdata)
#     LabeledArray{T,N}(newdata,newlabels)
# end
#
# Base.eltype{T}(::LabeledArray{T})=T
# Base.eltype{T}(::LabeledArray{T})=T
# Base.eltype{T}(::Type{LabeledArray{T}})=T
# Base.eltype{T,N}(::Type{LabeledArray{T,N}})=T
#
Base.getindex(A::Array,l::LabelList)=LabeledArray(A,l.labels)
Base.getindex(A::SubArray,l::LabelList)=LabeledArray(A,l.labels)
# if VERSION.minor >= 3
#     Base.getindex(A::SharedArray,l::LabelList)=LabeledArray(A,l.labels)
# end
Base.getindex(A::LabeledArray,l::LabelList)=LabeledArray(A.data,l.labels)
#
# Base.setindex!(A::Array,B::LabeledArray,l::LabelList)=TensorOperations.tensorcopy!(B.data,B.labels,A,l.labels)
# Base.setindex!(A::SubArray,B::LabeledArray,l::LabelList)=TensorOperations.tensorcopy!(B.data,B.labels,A,l.labels)
# # if VERSION.minor >= 3
# #     Base.setindex!(A::SharedArray,B::LabeledArray,l::LabelList)=TensorOperations.tensorcopy!(B.data,B.labels,A,l.labels)
# # end
#
# # addition of arrays
# +(A::LabeledArray,B::LabeledArray)=LabeledArray(TensorOperations.tensoradd(A.data,A.labels,B.data,B.labels,A.labels),A.labels)
#
# # complex conjugation
# Base.conj(A::LabeledArray)=LabeledArray(conj(A.data),A.labels)
#
# # multiplication with scalars
# Base.scale(A::LabeledArray,a::Number)=LabeledArray(scale(A.data,a),A.labels)
# *(t::LabeledArray,a::Number)=scale(t,a)
# *(a::Number,t::LabeledArray)=scale(t,a)
# /(t::LabeledArray,a::Number)=scale(t,one(a)/a)
# \(a::Number,t::LabeledArray)=scale(t,one(a)/a)
#
# # general contraction
# *(A::LabeledArray,B::LabeledArray)=LabeledArray(TensorOperations.tensorcontract(A.data,A.labels,B.data,B.labels),symdiff(A.labels,B.labels))
#
# TensorOperations.scalar{T}(C::LabeledArray{T,0})=C.data[1]
#
# end
