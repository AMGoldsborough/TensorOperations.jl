import Base: +, -, *

immutable Indices{I}
end

abstract AbstractIndexedObject
indices(a::AbstractIndexedObject) = indices(typeof(a))

include("indexnotation/indexedobject.jl")
include("indexnotation/sum.jl")
include("indexnotation/product.jl")

macro tensor(arg)
    tensorify(arg)
end

function tensorify(ex::Expr)
    if ex.head == :(=) || ex.head == :(+=) || ex.head == :(-=)
        lhs = ex.args[1]
        rhs = ex.args[2]
        if isa(lhs,Expr) && lhs.head == :ref
            dst = tensorify(lhs.args[1])
            src = tensorify(rhs)
            indices = makeindex_expr(lhs)
            value = ex.head == :(+=) ? +1 : ex.head == :(-=) ? -1 : 0
            return :(deindexify!($dst, $src, $indices, $value))
        end
    end
    if ex.head == :(:=)
        lhs = ex.args[1]
        rhs = ex.args[2]
        if isa(lhs,Expr) && lhs.head == :ref
            dst = tensorify(lhs.args[1])
            src = tensorify(rhs)
            indices = makeindex_expr(lhs)
            return :($dst = deindexify($src, $indices))
        end
    end
    if ex.head == :ref
        indices = makeindex_expr(ex)
        t = tensorify(ex.args[1])
        return :(indexify($t,$indices))
    end
    return Expr(ex.head,map(tensorify,ex.args)...)
end
tensorify(ex::Symbol) = esc(ex)
tensorify(ex) = ex

function makeindex_expr(ex::Expr)
    if ex.head == :ref
        for i = 2:length(ex.args)
            isa(ex.args[i],Int) || isa(ex.args[i],Symbol) || isa(ex.args[i],Char) || error("cannot make indices from $ex")
        end
    else
        error("cannot make indices from $ex")
    end
    return :(Indices{$(tuple(ex.args[2:end]...))}())
end
