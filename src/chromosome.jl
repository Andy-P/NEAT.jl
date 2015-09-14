# Temporary workaround - default settings
#node_gene_type = genome.NodeGene
# conn_gene_type = genome.ConnectionGene

type Chromosome
    id::Int64
    inputCnt::Int64
    outputCnt::Int64
    node_gene_type::Symbol
    conn_gene_type::Symbol
    connection_genes::Dict{(Int64,Int64),ConnectionGene}
    node_genes::Vector{NodeGene}
    fitness::Float64
    species_id::Int64
    parent1_id::Int64
    parent2_id::Int64
    # a chromosome for general recurrent neural networks.
    function Chromosome(g::Global, parent1_id::Int64, parent2_id::Int64, node_gene_type::Symbol, conn_gene_type::Symbol)

        new(incChromeId!(g),
            g.cg.input_nodes, g.cg.output_nodes,

            # the type of NodeGene and ConnectionGene the chromosome carries
            node_gene_type, conn_gene_type,

            # how many genes of the previous type the chromosome has
            Dict{(Int64,Int64),ConnectionGene}(), # dictionary of connection genes
            [], # empty array of node_genes
            0., # stub for fitness function
            0, # species_id

            # parents ids: helps in tracking chromosome's genealogy
            parent1_id, parent2_id)
    end
end

function incChromeId!(g::Global)
    g.chromosomeCnt += 1
    return g.chromosomeCnt
end

function mutate(ch::Chromosome)
    # Mutates the chromosome
    if rand() < Config.prob_addnode
        ch.mutate_add_node!()
    elseif rand() < Config.prob_addconn
        ch.mutate_add_connection!()
    else
        map(cg -> cg.mutate(),ch.connection_genes) # mutate weights
        map(ng -> ng.mutate(),ch.node_genes[ch.inputCnt+1:end]) # mutate bias, response, and etc...
    end
end


function crossover(self::Chromosome, other::Chromosome)
    # Crosses over parents' chromosomes and returns a child

    # This can't happen! Parents must belong to the same species.
#         assert self.species_id == other.species_id, 'Different parents species ID: %d vs %d' \
#                                                          % (self.species_id, other.species_id)

#         # TODO: if they're of equal fitnesses, choose the shortest
        if self.fitness > other.fitness
            parent1 = self
            parent2 = other
        else
            parent1 = other
            parent2 = self
        end

#         # creates a new child
#         child = self.__class__(self.id, other.id, self._node_gene_type, self._conn_gene_type)

#         child._inherit_genes(parent1, parent2)

#         child.species_id = parent1.species_id
#         #child._input_nodes = parent1._input_nodes

#         return child
end

function inherit_genes(child, parent1, parent2)
    # Applies the crossover operator.
#     @assert(parent1.fitness >= parent2.fitness)

#     # Crossover connection genes
#     for cg1 in parent1.connection_genes
#         if haskey(parent2.connection_gene,cg1.key)
#             cg2 = parent2._connection_genes[cg1.key]
#         except KeyError:
#             # Copy excess or disjoint genes from the fittest parent
#             child._connection_genes[cg1.key] = cg1.copy()
#         else
#             if cg2.is_same_innov(cg1) # Always true for *global* INs
#                 # Homologous gene found
#                 new_gene = cg1.get_child(cg2)
#                 #new_gene.enable() # avoids disconnected neurons
#             else
#                 new_gene = cg1.copy()
#             end
#         end
#     end
#             child._connection_genes[new_gene.key] = new_gene

#         # Crossover node genes
#         for i, ng1 in enumerate(parent1._node_genes):
#             try:
#                 # matching node genes: randomly selects the neuron's bias and response
#                 child._node_genes.append(ng1.get_child(parent2._node_genes[i]))
#             except IndexError:
#                 # copies extra genes from the fittest parent
#                 child._node_genes.append(ng1.copy())
end


function mutate_add_node!(ch::Chromosome, g::Global)
        # Choose a random connection to split
        ks = collect(keys(ch.connection_genes))
        conn_to_split = ch.connection_genes[ks[rand(1:length(ks))]]

        ng = NodeGene(length(ch.node_genes)+1,:HIDDEN, 0., 1., g.cg.nn_activation, 1.0)
        push!(ch.node_genes, ng)
        new_conn1, new_conn2 = split(g, conn_to_split, ng.id)
        ch.connection_genes[new_conn1.key] = new_conn1
        ch.connection_genes[new_conn2.key] = new_conn2
        return (ng, conn_to_split) # the return is only used in genome_feedforward
end


function mutate_add_connection!(ch::Chromosome, g::Global)
    # Only for recurrent networks
    total_possible = (length(ch.node_genes) - ch.inputCnt)  * length(ch.node_genes)
    println("total_possible $total_possible")
    remaining_conns = total_possible - length(ch.connection_genes)
    println("remaining_conns $remaining_conns")
    # Check if new connection can be added:
    if remaining_conns > 0
        n = rand(1:remaining_conns)
        count = 1
        # Count connections
        println("n $n")
        for in_node in ch.node_genes
            for out_node in ch.node_genes[ch.inputCnt+1:end]
                if !haskey(ch.connection_genes,(in_node.id, out_node.id)) # if fDree connection
                    if count == n # Connection to create
                        weight = randn() * g.cg.weight_stdev
                        cg = ConnectionGene(g, in_node.id, out_node.id, weight, true)
                        ch.connection_genes[cg.key] = cg
                        println(cg)
                        return
                    end
                    count += 1
                end
            end
        end
    end
end

# compatibility function
function distance(self::Chromosome, other::Chromosome)
    # Returns the distance between this chromosome and the other.
#     if len(self._connection_genes) > len(other._connection_genes):
#         chromo1 = self
#         chromo2 = other
#     else:
#         chromo1 = other
#         chromo2 = self

#     weight_diff = 0
#     matching = 0
#     disjoint = 0
#     excess = 0

#     max_cg_chromo2 = max(chromo2._connection_genes.values())

#     for cg1 in chromo1._connection_genes.values():
#         try:
#             cg2 = chromo2._connection_genes[cg1.key]
#         except KeyError:
#             if cg1 > max_cg_chromo2:
#                 excess += 1
#             else:
#                 disjoint += 1
#         else:
#             # Homologous genes
#             weight_diff += math.fabs(cg1.weight - cg2.weight)
#             matching += 1

#     disjoint += len(chromo2._connection_genes) - matching

#     #assert(matching > 0) # this can't happen
#     distance = Config.excess_coeficient * excess + \
#                Config.disjoint_coeficient * disjoint
#     if matching > 0:
#         distance += Config.weight_coeficient * (weight_diff/matching)

#     return distance
end

function size(ch::Chromosome)
    # Defines chromosome 'complexity': number of hidden nodes plus
    # number of enabled connections (bias is not considered)
    num_hidden = length(ch.node_genes) - ch.inputCnt - ch.outputCnt
    conns_enabled = sum(map(cg->ch.connection_genes[cg].enable==true? 1:0, collect(keys(ch.connection_genes))))

    return num_hidden, conns_enabled
end
#         return (num_hidden, conns_enabled)

#     def __cmp__(self, other)
#         """ First compare chromosomes by their fitness and then by their id.
#             Older chromosomes (lower ids) should be prefered if newer ones
#             performs the same.
#         """
#         #return cmp(self.fitness, other.fitness) or cmp(other.id, self.id)
#         return cmp(self.fitness, other.fitness)

#     def __str__(self)
#         s = "Nodes:"
#         for ng in self._node_genes:
#             s += "\n\t" + str(ng)
#         s += "\nConnections:"
#         connections = self._connection_genes.values()
#         connections.sort()
#         for c in connections:
#             s += "\n\t" + str(c)
#         return s

#     def add_hidden_nodes(self, num_hidden)
#         id = len(self._node_genes)+1
#         for i in range(num_hidden):
#             node_gene = self._node_gene_type(id,
#                                           nodetype = 'HIDDEN',
#                                           activation_type = Config.nn_activation)
#             self._node_genes.append(node_gene)
#             id += 1
#             # Connect all nodes to it
#             for pre in self._node_genes:
#                 weight = random.gauss(0, Config.weight_stdev)
#                 cg = self._conn_gene_type(pre.id, node_gene.id, weight, True)
#                 self._connection_genes[cg.key] = cg
#             # Connect it to all nodes except input nodes
#             for post in self._node_genes[self._input_nodes:]:
#                 weight = random.gauss(0, Config.weight_stdev)
#                 cg = self._conn_gene_type(node_gene.id, post.id, weight, True)
#                 self._connection_genes[cg.key] = cg

#     @classmethod
function create_unconnected(g::Global)

    # Creates a chromosome for an unconnected feedforward network with no hidden nodes.
    c = Chromosome(g, 0, 0, :NodeGene, :ConnectionGene)
    id = 1
    # Create node genes
    for i = 1:c.inputCnt
        push!(c.node_genes, NodeGene(id, :INPUT))
        id += 1
    end
#         #c._input_nodes += num_input
    for i in 1:c.outputCnt
        push!(c.node_genes, NodeGene(id, :OUTPUT))
        id += 1
    end
    @assert id == length(c.node_genes) + 1
    return c
end

function create_minimally_connected(g::Global)

    # Creates a chromosome for a minimally connected feedforward network with no hidden nodes.
    # That is, each output node will have a single connection from a randomly chosen input node.
    ch = create_unconnected(g)
    for node_gene in ch.node_genes[end-ch.outputCnt+1:end] # each output node
        # Connect it to a random input node
        input_node = ch.node_genes[rand(1:ch.inputCnt)]
        weight = randn() * g.cg.weight_stdev
        cg = ConnectionGene(g, input_node.id, node_gene.id, weight)
        ch.connection_genes[cg.key] = cg
    end
    return ch
end

function create_fully_connected(g::Global)

    # Creates a chromosome for a fully connected feedforward network with no hidden nodes.
    ch = create_unconnected(g)
    for node_gene in ch.node_genes[end-ch.outputCnt+1:end] # each output node
        # Connect it to all input nodes
        for input_node in ch.node_genes[1:ch.inputCnt]
            # TODO: review the initial weights distribution
            # weight = random.uniform(-1, 1)*Config.random_range
            weight = randn() * g.cg.weight_stdev
            cg = ConnectionGene(g, input_node.id, node_gene.id, weight)
            ch.connection_genes[cg.key] = cg
        end
    end
    return ch
end


# class FFChromosome(Chromosome)
#     """ A chromosome for feedforward neural networks. Feedforward
#         topologies are a particular case of Recurrent NNs.
#     """
#     def __init__(self, parent1_id, parent2_id, node_gene_type, conn_gene_type):
#         super(FFChromosome, self).__init__(parent1_id, parent2_id, node_gene_type, conn_gene_type)
#         self.__node_order = [] # hidden node order (for feedforward networks)

#     node_order = property(lambda self: self.__node_order)

#     def _inherit_genes(child, parent1, parent2):
#         super(FFChromosome, child)._inherit_genes(parent1, parent2)

#         child.__node_order = parent1.__node_order[:]

#         assert(len(child.__node_order) == len([n for n in child.node_genes if n.type == 'HIDDEN']))

#     def _mutate_add_node(self):
#         ng, split_conn = super(FFChromosome, self)._mutate_add_node()
#         # Add node to node order list: after the presynaptic node of the split connection
#         # and before the postsynaptic node of the split connection
#         if self._node_genes[split_conn.innodeid - 1].type == 'HIDDEN':
#             mini = self.__node_order.index(split_conn.innodeid) + 1
#         else:
#             # Presynaptic node is an input node, not hidden node
#             mini = 0
#         if self._node_genes[split_conn.outnodeid - 1].type == 'HIDDEN':
#             maxi = self.__node_order.index(split_conn.outnodeid)
#         else:
#             # Postsynaptic node is an output node, not hidden node
#             maxi = len(self.__node_order)
#         self.__node_order.insert(random.randint(mini, maxi), ng.id)
#         assert(len(self.__node_order) == len([n for n in self.node_genes if n.type == 'HIDDEN']))
#         return (ng, split_conn)

#     def _mutate_add_connection(self):
#         # Only for feedforwad networks
#         num_hidden = len(self.__node_order)
#         num_output = len(self._node_genes) - self._input_nodes - num_hidden

#         total_possible_conns = (num_hidden+num_output)*(self._input_nodes+num_hidden) - \
#             sum(range(num_hidden+1))

#         remaining_conns = total_possible_conns - len(self._connection_genes)
#         # Check if new connection can be added:
#         if remaining_conns > 0:
#             n = random.randint(0, remaining_conns - 1)
#             count = 0
#             # Count connections
#             for in_node in (self._node_genes[:self._input_nodes] + self._node_genes[-num_hidden:]):
#                 for out_node in self._node_genes[self._input_nodes:]:
#                     if (in_node.id, out_node.id) not in self._connection_genes.keys() and \
#                         self.__is_connection_feedforward(in_node, out_node):
#                         # Free connection
#                         if count == n: # Connection to create
#                             #weight = random.uniform(-Config.random_range, Config.random_range)
#                             weight = random.gauss(0,1)
#                             cg = self._conn_gene_type(in_node.id, out_node.id, weight, True)
#                             self._connection_genes[cg.key] = cg
#                             return
#                         else:
#                             count += 1

#     def __is_connection_feedforward(self, in_node, out_node):
#         return in_node.type == 'INPUT' or out_node.type == 'OUTPUT' or \
#             self.__node_order.index(in_node.id) < self.__node_order.index(out_node.id)

#     def add_hidden_nodes(self, num_hidden):
#         id = len(self._node_genes)+1
#         for i in range(num_hidden):
#             node_gene = self._node_gene_type(id,
#                                           nodetype = 'HIDDEN',
#                                           activation_type = Config.nn_activation)
#             self._node_genes.append(node_gene)
#             self.__node_order.append(node_gene.id)
#             id += 1
#             # Connect all input nodes to it
#             for pre in self._node_genes[:self._input_nodes]:
#                 weight = random.gauss(0, Config.weight_stdev)
#                 cg = self._conn_gene_type(pre.id, node_gene.id, weight, True)
#                 self._connection_genes[cg.key] = cg
#                 assert self.__is_connection_feedforward(pre, node_gene)
#             # Connect all previous hidden nodes to it
#             for pre_id in self.__node_order[:-1]:
#                 assert pre_id != node_gene.id
#                 weight = random.gauss(0, Config.weight_stdev)
#                 cg = self._conn_gene_type(pre_id, node_gene.id, weight, True)
#                 self._connection_genes[cg.key] = cg
#             # Connect it to all output nodes
#             for post in self._node_genes[self._input_nodes:(self._input_nodes + self._output_nodes)]:
#                 assert post.type == 'OUTPUT'
#                 weight = random.gauss(0, Config.weight_stdev)
#                 cg = self._conn_gene_type(node_gene.id, post.id, weight, True)
#                 self._connection_genes[cg.key] = cg
#                 assert self.__is_connection_feedforward(node_gene, post)

#     def __str__(self):
#         s = super(FFChromosome, self).__str__()
#         s += '\nNode order: ' + str(self.__node_order)
#         return s

# if __name__ == '__main__':
#     # Example
#     import visualize
#     # define some attributes
#     node_gene_type = genome.NodeGene         # standard neuron model
#     conn_gene_type = genome.ConnectionGene   # and connection link
#     Config.nn_activation = 'exp'             # activation function
#     Config.weight_stdev = 0.9                # weights distribution

#     Config.input_nodes = 2                   # number of inputs
#     Config.output_nodes = 1                  # number of outputs

#     # creates a chromosome for recurrent networks
#     #c1 = Chromosome.create_fully_connected()

#     # creates a chromosome for feedforward networks
#     c2 = FFChromosome.create_fully_connected()
#     # add two hidden nodes
#     c2.add_hidden_nodes(2)
#     # apply some mutations
#     #c2._mutate_add_node()
#     #c2._mutate_add_connection()

#     # check the result
#     #visualize.draw_net(c1) # for recurrent nets
#     visualize.draw_ff(c2)   # for feedforward nets
#     # print the chromosome
#     print  c2
