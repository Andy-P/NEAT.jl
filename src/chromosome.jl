abstract ChromoType
type Recurrent   <: ChromoType end
type FeedForward <: ChromoType end

type Chromosome
    id::Int64
    inputCnt::Int64
    outputCnt::Int64
    node_gene_type::ChromoType
    conn_gene_type::Symbol
    connection_genes::Dict{(Int64,Int64),ConnectionGene}
    node_genes::Vector{NodeGene}
    fitness::Float64
    species_id::Int64
    parent1_id::Int64
    parent2_id::Int64
    node_order::Vector{Int64}
    # a chromosome for general recurrent neural networks.
    function Chromosome(g::Global, parent1_id::Int64, parent2_id::Int64, node_gene_type::ChromoType, conn_gene_type::Symbol)

        new(incChromeId!(g),
            g.cf.input_nodes, g.cf.output_nodes,

            # the type of NodeGene and ConnectionGene the chromosome carries
            node_gene_type, conn_gene_type,

            # how many genes of the previous type the chromosome has
            Dict{(Int64,Int64),ConnectionGene}(), # dictionary of connection genes
            [], # empty array of node_genes
            0., # stub for fitness function
            0, # species_id

            # parents ids: helps in tracking chromosome's genealogy
            parent1_id, parent2_id,
            [] # node order (only used by feedforward networks)
            )
    end
end

function incChromeId!(g::Global)
    g.chromosomeCnt += 1
    return g.chromosomeCnt
end

function mutate(ch::Chromosome, g::Global)
    # Mutates the chromosome
    if rand() < g.cf.prob_addnode
        mutate_add_node!(ch, g, ch.node_gene_type)
    elseif rand() < g.cf.prob_addconn
        mutate_add_connection!(ch, g, ch.node_gene_type)
    else
        map(cg -> mutate!(cg[2], g.cf), ch.connection_genes) # mutate weights
        map(ng -> mutate!(ng, g.cf), ch.node_genes[ch.inputCnt+1:end]) # mutate bias, response, and etc...
    end
    return ch
end


function crossover(g::Global, self::Chromosome, other::Chromosome)
    # Crosses over parents' chromosomes and returns a child

    # This can't happen! Parents must belong to the same species.
    @assert self.species_id == other.species_id

    # TODO: if they're of equal fitnesses, choose the shortest
    parent1, parent2 = self.fitness >= other.fitness? (self,other):(other,self)

    # create a new child
    child = Chromosome(g, self.id, other.id, self.node_gene_type, self.conn_gene_type)
    inherit_genes!(g, child ,parent1, parent2)
    child.species_id = parent1.species_id
    child.fitness = parent1.fitness
    #child._input_nodes = parent1._input_nodes

    return child
end

function inherit_genes!(g::Global, child::Chromosome, parent1::Chromosome, parent2::Chromosome)
    # Applies the crossover operator.
    @assert parent1.fitness >= parent2.fitness

    # Crossover connection genes
    for (key, cg1) in parent1.connection_genes
        if haskey(parent2.connection_genes, key)
            cg2 = parent2.connection_genes[cg1.key]
            gene = is_same_innov(cg1, cg2)? deepcopy(get_child(cg1, cg2)) : deepcopy(cg1)
            child.connection_genes[cg1.key] = gene
        else # Copy excess or disjoint genes from the fittest parent
            child.connection_genes[cg1.key] = deepcopy(cg1)
        end
    end

    # Crossover node genes
    for i = 1:length(parent1.node_genes)
        ng1 = parent1.node_genes[i]
        # matching node genes: randomly selects the neuron's bias and response
        if length(parent2.node_genes) >= i && parent2.node_genes[i].id == ng1.id
            push!(child.node_genes, get_child(ng1, parent2.node_genes[i]))
        else
            # copies extra genes from the fittest parent
            push!(child.node_genes, deepcopy(ng1))
        end
    end

    child.node_order = parent1.node_order[:]
end


#----------  Recurrent Mutation  ----------

function mutate_add_node!(ch::Chromosome, g::Global,::Recurrent)
    # Choose a random connection to split
    ks = collect(keys(ch.connection_genes))
    toSpilt = rand(1:length(ks))
    conn_to_split = ch.connection_genes[ks[toSpilt]]

    ng = NodeGene(length(ch.node_genes)+1,:HIDDEN, 0., 1., g.cf.nn_activation, 1.0)
#     println("To Split: $toSpilt \n   $conn_to_split\n   $ng")
    push!(ch.node_genes, ng)
    new_conn1, new_conn2 = split(g, conn_to_split, ng.id)
    ch.connection_genes[new_conn1.key] = new_conn1
    ch.connection_genes[new_conn2.key] = new_conn2
    return (ng, conn_to_split) # the return is only used in genome_feedforward
end


function mutate_add_connection!(ch::Chromosome, g::Global, ::Recurrent)
    # Only for recurrent networks
    total_possible = (length(ch.node_genes) - ch.inputCnt)  * length(ch.node_genes)
    remaining_conns = total_possible - length(ch.connection_genes)

    # Check if new connection can be added:
    if remaining_conns > 0
        n = rand(1:remaining_conns)
        count = 1
        # Count connections
        for in_node in ch.node_genes
            for out_node in ch.node_genes[ch.inputCnt+1:end]
                if !haskey(ch.connection_genes,(in_node.id, out_node.id)) # if fDree connection
                    if count == n # Connection to create
                        weight = randn() * g.cf.weight_stdev
                        cg = ConnectionGene(g, in_node.id, out_node.id, weight, true)
                        ch.connection_genes[cg.key] = cg
#                         println(cg)
                        return
                    end
                    count += 1
                end
            end
        end
    end
end

#----------  FeedForward Mutation  ----------

function mutate_add_node!(ch::Chromosome, g::Global,::FeedForward)

    ng, split_conn = mutate_add_node!(ch::Chromosome, g::Global, Recurrent())

    if length(ch.node_order) == 0
        push!(ch.node_order, ng.id)
        return ng, split_conn
    end
    # Add node to node order list: after the presynaptic node of the split connection
    # and before the postsynaptic node of the split connection
    mini = ch.node_genes[split_conn.inId].ntype == :HIDDEN?
        findfirst(ch.node_order, split_conn.inId)+1:1

    maxi = ch.node_genes[split_conn.outId].ntype == :HIDDEN?
        findfirst(ch.node_order, split_conn.outId):length(ch.node_order)

    idx = mini <= maxi? rand(mini:maxi):mini # unnecessary?
    insert!(ch.node_order, idx, ng.id)
#     assert(length(ch.node_order) == length([n for n in ch.node_genes if n.type == :HIDDEN]))
    return ng, split_conn

end

function mutate_add_connection!(ch::Chromosome, g::Global,::FeedForward)

    # Only for feedforwad networks
    num_hidden = length(ch.node_order)
    num_output = length(ch.node_genes) - ch.inputCnt - num_hidden

    total_possible_conns = (num_hidden + num_output)*(ch.inputCnt + num_hidden) - sum([1:num_hidden+1])

    remaining_conns = total_possible_conns - length(ch.connection_genes)
    # Check if new connection can be added:
    if remaining_conns > 0
        n = rand(0:remaining_conns - 1)
        count = 0
        # Count connections
        for in_node in vcat(ch.node_genes[1:ch.inputCnt],ch.node_genes[end-num_hidden+1:end])
            for out_node in ch.node_genes[ch.inputCnt+1:end]
                if !haskey(ch.connection_genes,(in_node.id, out_node.id)) &&　is_connection_feedforward(ch, in_node, out_node)
                    # Free connection
                    if count == n # Connection to create
                        weight = randn() * g.cf.weight_stdev
                        cg = ConnectionGene(g, in_node.id, out_node.id, weight, true)
                        ch.connection_genes[cg.key] = cg
                        return
                    else
                        count += 1
                    end
                end
            end
        end
    end

end

function is_connection_feedforward(ch::Chromosome, in_node::NodeGene, out_node::NodeGene)
    return in_node.ntype == :INPUT || out_node.ntype == :OUTPUT ||
        findfirst(ch.node_order, in_node.id) < findfirst(ch.node_order, out_node.id)
end

#----------  End of Mutation  ----------

# compatibility function
function distance(self::Chromosome, other::Chromosome, cf::Config)

    # Returns the distance between this chromosome and the other.
    chromo1, chromo2 = length(self.connection_genes) > length(other.connection_genes)? (self,other):(other,self)

    weight_diff = 0
    matching = 0
    disjoint = 0
    excess = 0

    max_cg_chromo2 = maxInnov(chromo2.connection_genes)

    for (k, cg1) in chromo1.connection_genes
        if haskey(chromo2.connection_genes, k)
            # Homologous genes
            cg2 = chromo2.connection_genes[cg1.key]
            weight_diff += abs(cg1.weight - cg2.weight)
            matching += 1
        else
            if cg1.innovNumber > max_cg_chromo2.innovNumber
                excess += 1
            else
                disjoint += 1
            end
        end
    end

    disjoint += length(chromo2.connection_genes) - matching
    d = cf.excess_coeficient * excess + cf.disjoint_coeficient * disjoint

    return matching > 0? d + cf.weight_coeficient * weight_diff / matching : d
end

function size(ch::Chromosome)
    # Defines chromosome 'complexity': number of hidden nodes plus
    # number of enabled connections (bias is not considered)
    num_hidden = length(ch.node_genes) - ch.inputCnt - ch.outputCnt
    conns_enabled = sum(map(cg->ch.connection_genes[cg].enable==true? 1:0, collect(keys(ch.connection_genes))))

    return num_hidden, conns_enabled
end

#     def __cmp__(self, other)
#         """ First compare chromosomes by their fitness and then by their id.
#             Older chromosomes (lower ids) should be prefered if newer ones
#             performs the same.
#         """
#         #return cmp(self.fitness, other.fitness) or cmp(other.id, self.id)
#         return cmp(self.fitness, other.fitness)

function Base.show(io::IO, ch::Chromosome)
    s = "$(ch.node_gene_type) type\nNodes:\n"
    for ng in ch.node_genes
        s = "$(s)\t$(ng)"
    end
    s = "$(s)Order:\n\t$(ch.node_order)"

    s = "$(s)\nConnections:"
    ord = map(x->(ch.connection_genes[x].innovNumber,x), collect(keys(ch.connection_genes)))
    sort!(ord, by=x->x[1])
    for (innov, (inId ,outId)) in ord
        s = "$(s)\n\t$(ch.connection_genes[(inId ,outId)])"
    end
    @printf(io,"\n%6s", s)
    return
end

function add_hidden_nodes!(g::Global, ch::Chromosome, num_hidden::Int64, ::Recurrent)

    id = length(ch.node_genes)+1
    for i in 1:num_hidden
        node_gene = NodeGene(id, :HIDDEN, 0., 1., g.cf.nn_activation)
        push!(ch.node_genes, node_gene)
        id += 1
        # Connect all nodes to it
        for pre in ch.node_genes
            weight = randn() * g.cf.weight_stdev
            cg = ConnectionGene(g, pre.id, node_gene.id, weight)
            ch.connection_genes[cg.key] = cg
        end
        # Connect it to all nodes except input nodes
        for post in ch.node_genes[ch.inputCnt+1:end]
            weight = randn() * g.cf.weight_stdev
            cg = ConnectionGene(g, node_gene.id, post.id, weight)
            ch.connection_genes[cg.key] = cg
        end
    end
end

function create_unconnected(g::Global)

    # Creates a chromosome for an unconnected feedforward network with no hidden nodes.
    c = Chromosome(g, 0, 0, (g.cf.feedforward? FeedForward():Recurrent()), :ConnectionGene)

    id = 1
    # Create node genes
    for i = 1:c.inputCnt
        push!(c.node_genes, NodeGene(id, :INPUT, 0., 1., :none))
        id += 1
    end
#         #c._input_nodes += num_input
    for i in 1:c.outputCnt
        push!(c.node_genes, NodeGene(id, :OUTPUT, 0., 1., g.cf.nn_activation))
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
        weight = randn() * g.cf.weight_stdev
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
            weight = randn() * g.cf.weight_stdev
            cg = ConnectionGene(g, input_node.id, node_gene.id, weight)
            ch.connection_genes[cg.key] = cg
        end
    end
    return ch
end

function add_hidden_nodes!(g::Global, ch::Chromosome, num_hidden::Int64, ::FeedForward)
    id = length(ch.node_genes)+1
    for i in 1:num_hidden
        node_gene = NodeGene(id, :HIDDEN, 0., 1., g.cf.nn_activation)
        push!(ch.node_genes, node_gene)
        push!(ch.node_order,node_gene.id)
        id += 1

        # Connect all input nodes to it
        for pre in ch.node_genes[1:ch.inputCnt]
            weight = randn() * g.cf.weight_stdev
            cg = ConnectionGene(g, pre.id, node_gene.id, weight, true)
            ch.connection_genes[cg.key] = cg
            @assert is_connection_feedforward(ch, pre, node_gene) == true
        end

        # Connect all previous hidden nodes to it
        # Comment: Makes for one wierd network! Omit hidden->hidden??
#         for pre_id in ch.node_order[1:end-1]
#             @assert pre_id != node_gene.id
#             weight = randn() * g.cf.weight_stdev
#             cg = ConnectionGene(g, pre_id, node_gene.id, weight, true)
#             ch.connection_genes[cg.key] = cg
#         end

        # Connect it to all output nodes
        for post in ch.node_genes[ch.inputCnt+1:(ch.inputCnt + ch.outputCnt)]
            @assert post.ntype == :OUTPUT
            weight = randn() * g.cf.weight_stdev
            cg = ConnectionGene(g, node_gene.id, post.id, weight, true)
            ch.connection_genes[cg.key] = cg
#             assert self.__is_connection_feedforward(node_gene, post)
        end
    end
end


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
