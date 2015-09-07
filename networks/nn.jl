
sigmoid(x::Float64) = 1.0/(1.0 + math.exp(-x))
sigmoid(x::Float64, response::Float64) = 1.0/(1.0 + math.exp(-x*response))


type Synapse
    source::Int64
    destination::Int64
    weight::Float64
    Synapse(source::Int64, destination::Int64, weight::Float64) = new(source, destination, weight)
end

incoming(s::Synapse) = s.source * s.weight

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
    function Neuron(neurontype::Symbol, id::Int64, activation::Symbol=:sigm,response::Float64=1.)
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
        output += s.incoming()
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
end

addNeuron(network::Network, neuron::Neuron) = push!(network.neurons,neuron)

addSynapse(network::Network, synapse::Synapse) = push!(network.synapses,synapse)

flush!(network::Network) = for n in network.neurons  n.output = 0. end


#     def __repr__(self):
#         return '%d nodes and %d synapses' % (len(self.__neurons), len(self.__synapses))

#     #def activate(self, inputs=[]):
#     #    if Config.feedforward:
#     #        return self.sactivate(inputs)
#     #    else:
#     #        return self.pactivate(inputs)

#     def sactivate(self, inputs=[]):
#         '''Serial (asynchronous) network activation method. Mostly
#            used  in classification tasks (supervised learning) in
#            feedforward topologies. All neurons are updated (activated)
#            one at a time following their order of importance, so if
#            you're defining your own feedforward topology, make sure
#            you got them in the right order of activation.
#         '''
#         assert len(inputs) == self._num_inputs, "Wrong number of inputs."
#         # assign "input neurons'" output values (sensor readings)

#         it = iter(inputs)
#         for n in self.__neurons[:self._num_inputs]:
#             if(n._type == 'INPUT'):
#                 n._output = it.next() # iterates over inputs
#         # activate all neurons in the network (except for the inputs)
#         net_output = []
#         for n in self.__neurons[self._num_inputs:]:
#             n._output = n.activate()
#             if(n._type == 'OUTPUT'):
#                 net_output.append(n._output)
#         return net_output

#     def pactivate(self, inputs=[]):
#         '''Parallel (synchronous) network activation method. Mostly used
#            for control and unsupervised learning (i.e., artificial life)
#            in recurrent networks. All neurons are updated (activated)
#            simultaneously.
#         '''
#         assert len(inputs) == self._num_inputs, "Wrong number of inputs."

#         # the current state is like a "photograph" taken at each time step
#         # reresenting all neuron's state at that time (think of it as a clock)
#         current_state = []
#         it = iter(inputs)
#         for n in self.__neurons:
#             if n._type == 'INPUT':
#                 n._output = it.next() # iterates over inputs
#             else: # hidden or output neurons
#                 current_state.append(n.activate())
#         # updates all neurons at once
#         net_output = []
#         for n, state in zip(self.__neurons[self._num_inputs:], current_state):
#             n._output = state # updates from the previous step
#             if n._type == 'OUTPUT':
#                 net_output.append(n._output)

#         return net_output

# class FeedForward(Network):
#     """ A feedforward network is a particular class of neural network.
#         Only one hidden layer is considered for now.
#     """

#     def __init__(self, layers, use_bias=False, activation_type=None):
#         super(FeedForward, self).__init__()

#         self.__input_layer   = layers[0]
#         self.__output_layer  = layers[-1]
#         self.__hidden_layers = layers[1:-1]
#         self.__use_bias = use_bias

#         self._num_inputs = layers[0]
#         self.__create_net(activation_type)

#     def __create_net(self, activation_type):

#         # assign random weights for bias
#         if self.__use_bias:
#             r = random.uniform
#         else:
#             r = lambda a,b: 0

#         for i in xrange(self.__input_layer):
#             self.add_neuron(Neuron('INPUT'))

#         for i in xrange(self.__hidden_layers[0]):
#             self.add_neuron(Neuron('HIDDEN', bias = r(-1,1),
#                                    response = 1,
#                                    activation_type = activation_type))

#         for i in xrange(self.__output_layer):
#             self.add_neuron(Neuron('OUTPUT', bias = r(-1,1),
#                                    response = 1,
#                                    activation_type = activation_type))

#         r = random.uniform  # assign random weights
#         # inputs -> hidden
#         for i in self.neurons[:self.__input_layer]:
#                 for h in self.neurons[self.__input_layer:-self.__output_layer]:
#                         self.add_synapse(Synapse(i, h, r(-1,1)))
#         # hidden -> outputs
#         for h in self.neurons[self.__input_layer:-self.__output_layer]:
#                 for o in self.neurons[-self.__output_layer:]:
#                         self.add_synapse(Synapse(h, o, r(-1,1)))

# def create_phenotype(chromo):
#         """ Receives a chromosome and returns its phenotype (a neural network) """

#         neurons_list = [Neuron(ng._type, ng._id,
#                                ng._bias,
#                                ng._response,
#                                ng.activation_type)
#                         for ng in chromo._node_genes]

#         conn_list = [(cg.innodeid, cg.outnodeid, cg.weight)
#                      for cg in chromo.conn_genes if cg.enabled]

#         return Network(neurons_list, conn_list, chromo.sensors)

# def create_ffphenotype(chromo):
#     """ Receives a chromosome and returns its phenotype (a neural network) """

#     # first create inputs
#     neurons_list = [Neuron('INPUT', ng.id, 0, 0) \
#                     for ng in chromo.node_genes[:chromo.sensors] if ng.type == 'INPUT']

#     # Add hidden nodes in the right order
#     for id in chromo.node_order:
#         neurons_list.append(Neuron('HIDDEN',
#                                    id, chromo.node_genes[id-1].bias,
#                                    chromo.node_genes[id-1].response,
#                                    chromo.node_genes[id-1].activation_type))
#     # finally the output
#     neurons_list.extend(Neuron('OUTPUT', ng.id, ng.bias,
#                                ng.response, ng.activation_type) \
#                                for ng in chromo.node_genes if ng.type == 'OUTPUT')

#     assert(len(neurons_list) == len(chromo.node_genes))

#     conn_list = [(cg.innodeid, cg.outnodeid, cg.weight) \
#                  for cg in chromo.conn_genes if cg.enabled]

#     return Network(neurons_list, conn_list, chromo.sensors)

# if __name__ == "__main__":
#     # Example
#     #from neat import visualize

#     nn = FeedForward([2,10,3], use_bias=False, activation_type = 'exp')
#     ##visualize.draw_ff(nn)
#     print 'Serial activation method: '
#     for t in range(3):
#         print nn.sactivate([1,1])

#     #print 'Parallel activation method: '
#     #for t in range(3):
#         #print nn.pactivate([1,1])

#     # defining a neural network manually
#     #neurons = [Neuron('INPUT', 1), Neuron('HIDDEN', 2), Neuron('OUTPUT', 3)]
#     #connections = [(1, 2, 0.5), (1, 3, 0.5), (2, 3, 0.5)]

#     #net = Network(neurons, connections) # constructs the neural network
#     #visualize.draw_ff(net)
#     #print net.pactivate([0.04]) # parallel activation method
#     #print net # print how many neurons and synapses our network has
