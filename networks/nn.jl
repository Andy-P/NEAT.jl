
import Base.tanh
sigm(x::Float64, γ::Float64=1.) = 1.0/(1.0 + exp(-x*γ))
tanh(x::Real, γ::Real) = (exp(2x*γ)-1) / (exp(2x*γ)+1)
relu(x::Real) = max(zero(eltype(x)),x)

# type Synapse
#     source::Neuron
#     destination::Neuron
#     weight::Float64
#     Synapse(source::Neuron, destination::Neuron, weight::Float64) = new(source, destination, weight)
# end

# incoming(s::Synapse) = s.weight * s.source
type Synapse
    source
    destination
    weight::Float64
    function Synapse(source, destination, weight::Float64)
        s = new(source, destination, weight);
        addSynapse(destination, s); s
    end
end

incoming(s::Synapse) = s.weight * s.source.output

function Base.show(io::IO, s::Synapse)
    @printf(io,"%d -> %3.5f -> %d\n", s.source.id, s.weight, s.destination.id)
end

type Neuron
    id::Int64
    synapses::Vector{Synapse}
    bias::Float64
    nType::Symbol       # [:INPUT, :OUTPUT, :HIDDEN]
    activation::Function  # [:sigm, :tanh, :relu]
    response::Float64   # default = 4.924273 (Stanley, p. 146)
    output::Float64     # for recurrent networks all neurons must have an "initial state"
    function Neuron(neurontype::Symbol, id::Int64, bias::Float64=0., activation::Symbol=:sigm, γ::Float64=1.)
        f = activation ==:sigm?  x->sigm(x,γ):activation ==:tanh? x->tanh(x,γ):x->relu(x)
        new(id,[],bias, neurontype, f, γ, 0.)
    end
end


addSynapse(n::Neuron, s::Synapse) = push!(n.synapses,s)   # adds the synapse to the destination neuron

function activate(n::Neuron)
#         "Activates the neuron"
        @assert n.nType != :INPUT
        return n.activation(updateActivation!(n) + n.bias)
end

function updateActivation!(n::Neuron)
    output = 0.
    for s in n.synapses
        output += incoming(s)
    end
    return output
end

# "Prints neuron's current state"
function currentOutput(n::Neuron)
    println("state: $(n.id) - output: $(n.output)")
    return n.output
end

function Base.show(io::IO, n::Neuron)
    @printf(io,"%%d %s\n", n.id, n.nType)
end
#     def __repr__(self):
#         return '%d %s' %(self._id, self._type)

type Network
    neurons::Vector{Neuron}
    synapses::Vector{Synapse}
    numInputs::Int64
    nntype::ChromoType
    Network(numInputs::Int64) = new([],[], numInputs, Recurrent())
    Network(neurons::Vector{Neuron}, synapses::Vector{Synapse}, numInputs::Int64, nntype::ChromoType) =
        new(neurons, synapses, numInputs, nntype)
end

addNeuron(network::Network, neuron::Neuron) = push!(network.neurons,neuron)

addSynapse(network::Network, synapse::Synapse) = push!(network.synapses,synapse)

flush!(network::Network) = for n in network.neurons  n.output = 0. end

function Base.show(io::IO, net::Network)
    @printf(io,"%s %d nodes and %d synapses\n",net.nntype, length(net.neurons), length(net.synapses))
end


function activate(::FeedForward, nn::Network, inputs::Vector)

    #=  Serial (asynchronous) network activation method. Mostly
    used  in classification tasks (supervised learning) in
    feedforward topologies. All neurons are updated (activated)
    one at a time following their order of importance, so if
    you're defining your own feedforward topology, make sure
    you got them in the right order of activation. =#

    @assert length(inputs) == nn.numInputs

    for i  = 1:nn.numInputs
        nn.neurons[i].output = inputs[i] # iterates over inputs
    end

    # activate all neurons in the network (except for the inputs)
    netOutput = zeros(0)
    for n in nn.neurons[nn.numInputs+1:end]
        n.output = activate(n)
        if n.nType == :OUTPUT  push!(netOutput,n.output) end
    end

    return netOutput

end

function activate(::Recurrent, nn::Network, inputs::Vector)

    #= Parallel (synchronous) network activation method. Mostly used
    for control and unsupervised learning (i.e., artificial life)
    in recurrent networks. All neurons are updated (activated)
    simultaneously. =#

    @assert length(inputs) == nn.numInputs

    # the current state is like a "photograph" taken at each time step
    # reresenting all neuron's state at that time (think of it as a clock)
    inputCnt = 0
    currentState = zeros(0)
    for n in nn.neurons
        if n.nType == :INPUT
            inputCnt += 1
            n.output = inputs[inputCnt]
        else
            push!(currentState,activate(n))
        end
    end

    netOutput = zeros(0)
    for i = nn.numInputs+1:length(nn.neurons)
        n = nn.neurons[i]
        n.output = currentState[i-nn.numInputs]
        if n.nType == :OUTPUT
            push!(netOutput,n.output)
        end
    end

    return netOutput
end


function createPhenotype(ch::Chromosome)
    # Receives a chromosome and returns its phenotype (a neural network)

    neurons = Neuron[]
    Idx2Neuron = Dict{Int64,Neuron}()
    if ch.node_gene_type == Recurrent()

        for ng in ch.node_genes
            n = Neuron(ng.ntype, ng.id, ng.bias, ng.activation, ng.response)
            push!(neurons, n)
            Idx2Neuron[ng.id] = n
        end

    else # FeedForward() type

        # first create inputs
        for ng in ch.node_genes[1:ch.inputCnt]
            n = Neuron(ng.ntype, ng.id, ng.bias, ng.activation, ng.response)
            push!(neurons,n)
            Idx2Neuron[ng.id] = n
        end

        # Add hidden nodes in the right order
        for id in ch.node_order
            ng = ch.node_genes[id]
            n = Neuron(ng.ntype, ng.id, ng.bias, ng.activation, ng.response)
            push!(neurons, n)
            Idx2Neuron[ng.id] = n
        end

        # finally the output
        for ng in ch.node_genes[ch.inputCnt+1:ch.inputCnt+ch.outputCnt]
            n = Neuron(ng.ntype, ng.id, ng.bias, ng.activation, ng.response)
            push!(neurons, n)
            Idx2Neuron[ng.id] = n
        end

    end

    @assert length(neurons) == length(ch.node_genes)

    synapses = Synapse[]
    for (k,cg) in ch.connection_genes
        if cg.enable push!(synapses, Synapse(Idx2Neuron[cg.inId], Idx2Neuron[cg.outId], cg.weight)) end
    end

    return Network(neurons, synapses, ch.inputCnt, ch.node_gene_type)

end


###################################
#         previous version        #
###################################

# import Base.+, Base.tanh

# sigm(x::Real) = 1.0 / (1.0 + exp(-x))
# sigm(x::Real, γ::Real) = 1.0 / (1.0 + exp(-γ*x))
# tanh(x::Real, γ::Real) = (exp(2x*γ)-1) / (exp(2x*γ)+1)
# relu(x::Real) = max(zero(x),x)



# abstract Gene

# type GeneConnection <: Gene
#     in::Gene
#     out::Gene
#     f::Function
#     w::Float64
#     expressed::Bool
#     inId::Int64
#     outId::Int64
#     id::Int64
#     function GeneConnection(inNode, inID, outID, id)
#        (gene = new(inNode,inNode,x->x,randn(),true,inID,outID,id);gene.f=()->gene.in.f()*gene.w;gene)
#     end
# end

# type BiasNode <: Gene
#     f::Function # always returns 1.0
#     BiasNode() = (g = new(y->y); g.f=()->1.; g)
# end

# type InputNode <: Gene
#     f::Function # always returns value of x
#     x::Float64
#     InputNode(value::Float64) = (g = new(y->y,value); g.f=()->g.x; g)
#     InputNode() = InputNode(0.)
# end

# +(x1::Float64,x2::GeneConnection) = x1 + x2.f() # needed for reduce(+,...)
# +(x1::GeneConnection,x2::GeneConnection) = x1.f() + x2.f()

# type GeneNode <: Gene
#     xs::Vector{GeneConnection}
#     bias::Float64
#     f::Function
#     g::Function # usually either a sigmoid, tanh, or relu function
#     expressed::Bool
#     id::Int64
#     function GeneNode(id::Int64, xs::Vector{GeneConnection}, bias::Float64)
#        (gene = new(xs,bias,x->x,x->relu(x),true,id); # default to rectified linear unit
#         gene.f=()-> gene.g(reduce(+, 0., gene.xs) + gene.bias);
#         map(x->x.out=gene,xs);
#         gene)
#     end
#     GeneNode(id::Int64, bias::Float64) = GeneNode(id, Array(NEAT.GeneConnection,0), bias)
#     GeneNode(id::Int64) = GeneNode(id, Array(NEAT.GeneConnection,0), 0.)
# end

# type Genome
#     numIns::Int64
#     numOuts::Int64
#     outputs::Vector{Gene}
#     inputs::Vector{Gene}
#     nodes::Vector{GeneNode}
#     connections::Vector{GeneConnection}
# end

# # # Initialize Network
# function initGenome(params::NeatParams)

# end

# function addNode(g::Genome, connIdx::Int64)


#     # get reference to the out node
#     c = g.connections[connIdx]
#     out = c.out

#     # remove from out node's forward pass function
# #     idx = findfirst(o->o.id==connIdx, out)

# #     c.expressed = false
#     # add to output functions
#     return out

# end


