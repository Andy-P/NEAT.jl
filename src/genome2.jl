# type of forward prop
#    1. Forward - no recrusivity (straight forward approach) ✔︎
#    2. Forward - with self-recrusive (only recursively connect to self) ✔︎
#    3. Forward - CTRnn recursivity allowed but shouldn't (?) create loops
#    4. Parallel - Standard but with no limits on structure
#    5. Parallel - CTRnn

# Problems/Requirements ✔︎
#   1.  If forward prop called, may get caught in loop
#   1.  Require some of the types to allow for backprop
#   2.  How to properly dispatch on node type
#       - need a subtype
#   3.  Parallel update cannot use chain pull method


# types: Forward, Forward -
# approach: Forward  = standard forward prop
#           Parallel = use previous output
#           Hyprid   = calls function based on type dispatch

# Clarity:  Inputs to a node are either...
#               Forward: Inputs that have just been calculated
#               Parallel: Inputs caluclated in previous step (AKA outputs)

import Base.+, Base.tanh

sigm(x::Real) = 1.0 / (1.0 + exp(-x))
sigm(x::Real, γ::Real) = 1.0 / (1.0 + exp(-γ*x))
tanh(x::Real, γ::Real) = (exp(2x*γ)-1) / (exp(2x*γ)+1)
relu(x::Real) = max(zero(x),x)

abstract Gene

type GeneConnection <: Gene
    in::Gene
    out::Gene
    f::Function
    w::Float64
    expressed::Bool
    inId::Int64
    outId::Int64
    id::Int64
    function GeneConnection(inNode, inID, outID, id)
       (gene = new(inNode,inNode,x->x,randn(),true,inID,outID,id);gene.f=()->gene.in.f()*gene.w;gene)
    end
end

type BiasNode <: Gene
    f::Function # always returns 1.0
    BiasNode() = (g = new(y->y); g.f=()->1.; g)
end

type InputNode <: Gene
    f::Function # always returns value of x
    x::Float64
    InputNode(value::Float64) = (g = new(y->y,value); g.f=()->g.x; g)
    InputNode() = InputNode(0.)
end

+(x1::Float64,x2::GeneConnection) = x1 + x2.f() # needed for reduce(+,...)
+(x1::GeneConnection,x2::GeneConnection) = x1.f() + x2.f()

type GeneNode <: Gene
    xs::Vector{GeneConnection}
    bias::Float64
    f::Function
    g::Function # usually either a sigmoid, tanh, or relu function
    expressed::Bool
    id::Int64
    function GeneNode(id::Int64, xs::Vector{GeneConnection}, bias::Float64)
       (gene = new(xs,bias,x->x,x->relu(x),true,id); # default to rectified linear unit
        gene.f=()-> gene.g(reduce(+, 0., gene.xs) + gene.bias);
        map(x->x.out=gene,xs);
        gene)
    end
    GeneNode(id::Int64, bias::Float64) = GeneNode(id, Array(NEAT.GeneConnection,0), bias)
    GeneNode(id::Int64) = GeneNode(id, Array(NEAT.GeneConnection,0), 0.)
end

type Genome
    numIns::Int64
    numOuts::Int64
    outputs::Vector{Gene}
    inputs::Vector{Gene}
    nodes::Vector{GeneNode}
    connections::Vector{GeneConnection}
end

# # Initialize Network
function initGenome(params::NeatParams)

end

function addNode(g::Genome, connIdx::Int64)


    # get reference to the out node
    c = g.connections[connIdx]
    out = c.out

    # remove from out node's forward pass function
#     idx = findfirst(o->o.id==connIdx, out)

#     c.expressed = false
    # add to output functions
    return out

end

