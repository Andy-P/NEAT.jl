type FeedForward end
fforward = FeedForward()

type Recurrent end
recurrent = Recurrent()

sigmoid(x::Float64) = 1.0/(1.0 + math.exp(-x))
sigmoid(x::Float64, response::Float64) = 1.0/(1.0 + math.exp(-x*response))


type Synapse
    source::Int64
    destination::Int64
    weight::Float64
    Synapse(source::Int64, destination::Int64, weight::Float64) = new(source, destination, weight)
end

incoming(s::Synapse) = s.weight * s.source

#     def __repr__(self):
#         return '%s -> %s -> %s' %(self._source._id, self._weight, self._destination._id)

type Neuron
    id::Int64
    synapses::Vector{Synapse}
    bias::Float64
    nType::Symbol       # [:INPUT, :OUTPUT, :HIDDEN]
    activation::Function  # [:sigm, :tahn, :relu]
    response::Float64   # default = 4.924273 (Stanley, p. 146)
    output::Float64     # for recurrent networks all neurons must have an "initial state"
    function Neuron(neurontype::Symbol, id::Int64, bias::Float64=0., activation::Symbol=:sigm,response::Float64=1.)
        new(id,[],bias, neurontype, x->sigmoid(x), response,0)
    end
end

function activate(n::Neuron)
#         "Activates the neuron"
#         assert self._type is not 'INPUT'
        return n.activation(updateActivation(n) + n.bias)
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

addSynapse(n::Neuron, s::Synapse) = push!(n.synapses,s)

#     def __repr__(self):
#         return '%d %s' %(self._id, self._type)

type Network
    neurons::Vector{Neuron}
    synapses::Vector{Synapse}
    numInputs::Int64
    Network(numInputs::Int64) = new([],[],　numInputs)
    Network(neurons::Vector{Neuron},synapses::Vector{Synapse},numInputs::Int64) = new(neurons,synapses,numInputs)
end

addNeuron(network::Network, neuron::Neuron) = push!(network.neurons,neuron)

addSynapse(network::Network, synapse::Synapse) = push!(network.synapses,synapse)

flush!(network::Network) = for n in network.neurons  n.output = 0. end

#     def __repr__(self):
#         return '%d nodes and %d synapses' % (len(self.__neurons), len(self.__synapses))


function activate(::FeedForward, network::Network, inputs::Vector)

    #=  Serial (asynchronous) network activation method. Mostly
    used  in classification tasks (supervised learning) in
    feedforward topologies. All neurons are updated (activated)
    one at a time following their order of importance, so if
    you're defining your own feedforward topology, make sure
    you got them in the right order of activation. =#

    @assert length(inputs) == n.numInputs

    for i  = 1:length(numInputs)
        network.neuron[i].output = inputs[i] # iterates over inputs
    end

    # activate all neurons in the network (except for the inputs)
    netOutput = zeros(0)
    for n in network.neuron[n.numInputs+1:end]
        n.output = n.activate()
        if n._type == :OUTPUT  push!(netOutput,n._output) end
    end

    return net_output

end

function activate(::Recurrent, network::Network, inputs::Vector)

    #= Parallel (synchronous) network activation method. Mostly used
    for control and unsupervised learning (i.e., artificial life)
    in recurrent networks. All neurons are updated (activated)
    simultaneously. =#

    @assert length(inputs) == n.numInputs

    # the current state is like a "photograph" taken at each time step
    # reresenting all neuron's state at that time (think of it as a clock)
    inputCnt = 0
    currentState = zeros(0)
    for n in network.neuron
        if n.ntype == :INPUT
            inputCnt += 1
            n.output = inputs[inputCnt]
        else
            push!(currentState,n.activate())
        end
    end

    netOutput = zeros(0)
    for i = numInputs+1:length(network.neuron)
        n = network.neuron[i]
        n.output = currentState[i-numInputs]
        if n.ntype == :OUTPUT
            push1(netOutput,n.output)
        end
    end

    return netOutput
end


function createPhenotype(chromo)
        # Receives a chromosome and returns its phenotype (a neural network)

    neurons = Array{Neuron,1}[]
    for ng in chromo.nodeGenes
        push!(neurons, Neuron(ng.nType, ng.id, ng.bias, ng.activation, ng.response))
    end

    synapses = Array{Synapse,1}[]
    for cg in chromo.connGenes
        push!(neurons, Synapse(cg.innodeid, cg.outnodeid, cg.weight))
    end

    return Network(neurons, synapses,length(chromo.sensors))

end
