module TensorOperations

export tensorcopy, tensoradd, tensortrace, tensorcontract, tensorproduct, scalar
export tensorcopy!, tensoradd!, tensortrace!, tensorcontract!, tensorproduct!

# LabelError
#------------
# Tensor operations, either using methods or via index notation, are specified
# by assigning labels to the different indices of an array. All errors related
# to invalid label configurations.
type LabelError <: Exception
    msg::String
end

# Constants that define base case in recursive algorithms
# ---------------------------------------------------------
const BASELENGTH=1024
# total number of elements involved in the base algorithms acting on the individual blocks

# Scalar
#--------
scalar(C::StridedArray) = length(C)==1 ? C[1] : throw(DimensionMismatch())

# Auxiliary functions
#---------------------
include("axpby.jl")
include("meta.jl")
include("similar.jl")
include("strideddata.jl")

# Tensor Operations
#-------------------
include("tensoradd.jl")
include("tensortrace.jl")
include("tensorcontract.jl")

# Index notation
#----------------
# include("indexnotation.jl")

end # module

# Index Notation
#----------------
# include("indexnotation.jl")
