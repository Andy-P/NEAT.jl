type NodeGene
    #= A node gene encodes the basic artificial neuron model.
    nodetype should be "INPUT", "HIDDEN", or "OUTPUT" =#
    id::Int64
    ntype::Symbol
    bias::Float64
    response::Float64
    activation::Symbol
    function NodeGene(id::Int64, nodetype::Symbol, bias::Float64=0., response::Float64=4.924273, activation::Symbol=:sigm)
        new(id, nodetype, bias, response, activation)
    end
end

function Base.show(io::IO, ng::NodeGene)
    @printf(io, "Node %5d %6s, bias %+2.10s, response %+2.10s\n", ng.id, ng.ntype, ng.bias, ng.response)
    return
end

function get_child(self::NodeGene, other::NodeGene)
    # Creates a new NodeGene ramdonly inheriting its attributes from parents
    @assert(self.id == other.id)

    return NodeGene(self.id, self.ntype,
        randbool() ? self.bias : other.bias,
        randbool() ? self.response : other.response,
        self.activation)
end

function mutate_bias(ng::NodeGene, cg::Config)
    ng.bias += randn() * cg.bias_mutation_power
    if (ng.bias > cg.max_weight)
        ng.bias = cg.max_weight
    elseif (ng.bias < cg.min_weight)
        ng.bias = cg.min_weight
    end
end

function mutate_response(ng::NodeGene, cg::Config)
    #  Mutates the neuron's average firing response.
    self.response += randn() * cg.bias_mutation_power
end

function copy(ng::NodeGene)
    return NodeGene(ng.id, ng.ntype, ng.bias, ng.response, ng.activation)
end

function mutate(ng::NodeGene, cg::Config)
    if rand() < cg.prob_mutatebias ng.mutate_bias(ng, cg) end
    if rand() < cg.prob_mutatebias ng.mutate_response(ng, cg) end
end

type CTNodeGene
    # Continuous-time node gene - used in CTRNNs.
    # The main difference here is the addition of
    # a decay rate given by the time constant.
    id::Int64
    ntype::Symbol
    bias::Float64
    response::Float64
    activation::Symbol
    timeConstant::Float64
    function NodeGene(id::Int64, nodetype::Symbol, bias::Float64=0, response::Float64=4.924273,
                      activation::Symbol=:sigm, timeConstant=1.0)
        new(id, nodetype, bias, response, activation, timeConstant)
    end
end

function mutate(ng::CTNodeGene, cf::Config)
    if rand() < cf.prob_mutatebias ng.mutate_bias(ng, cf) end
    if rand() < cf.prob_mutatebias ng.mutate_response(ng, cf) end
#     if rand() < 0.1 ng.mutate_time_constant() end
end

function mutate_time_constant(ng::CTNodeGene, cf::Config)
    # Warning: pertubing the time constant (tau) may result in numerical instability
    ng.timeConstant += randn() * .001
    if ng.timeConstant > cf.max_weight
        ng.timeConstant = cf.max_weight
    elseif ng.timeConstant < cf.min_weight
        ng.timeConstant = cf.min_weight
    end
end


function get_child(ng::CTNodeGene, other::CTNodeGene)
    # Creates a new NodeGene ramdonly inheriting its attributes from parents
    assert(ng.id == other.id)
    ng = CTNodeGene(ng.id, ng.ntype,
                    randbool()? ng.bias : other.bias,
                    randbool()? ng.response : other.response, self.activation,
                    randbool()? ng.timeConstant : other.timeConstant)
    return ng
end

function copy(ng::CTNodeGene)
    return CTNodeGene(ng.id, ng.ntype, ng.bias, ng.response, ng.activation, ng.timeConstant)
end

function Base.show(io::IO, ng::CTNodeGene)
    @printf(io, "Node %2d %6s, bias %+2.10s, response %+2.10s, activation %s, time constant %+2.5s\n",
            ng.id, ng.ntype, ng.bias, ng.response,ng.activation,ng.timeConstant)
    return
end

global_innov_number = 0
innovations = Dict{(Int64,Int64),Int64}() # global dictionary

function get_new_innov_number()
    global_innov_number += 1
    return global_innov_number
end

type ConnectionGene
    inId::Int64
    outId::Int64
    weight::Float64
    enabled::Bool
    key::(Int64,Int64)
    innovNumber::Int64
    function ConnectionGene(inId::Int64, outId::Int64, weight::Float64, enabled::Bool, innov::Int64=0)
        key = (inId, outId)
        if innov == 0
            if haskey(innovations,key)
                innovNumber = innovations[key]
            else
                innovNumber = get_new_innov_number()
                innovations[key] = innovNumber
            end
        else
            innovNumber = innov
        end
        new(inId, outId, weight, enabled, key, innovNumber)
    end
end

function mutate(cg::ConnectionGene, cf::Config)

    if rand() < cf.prob_mutate_weight
        mutate_weight(cg, cf)
    end

    if rand() <  cf.prob_togglelink
        cg.enable = true
    end

end

function mutate_weight(cg::ConnectionGene, cf::Config)
#     cg.weight += (rand() * 2 -1) * cf.weight_mutation_power
    cg.weight += randn() * cf.weight_mutation_power

    if cg.weight > Config.max_weight
        cg.weight = Config.max_weight
    elseif cg.weight < Config.min_weight
        cg.weight = Config.min_weight
    end
end

function weight_replaced(cg::ConnectionGene, cf::Config)
        # cg.weight = random.uniform(-Config.random_range, Config.random_range)
        cg.weight = randn() * cf.weight_stdev
end

function Base.show(io::IO, cg::ConnectionGene)
    @printf(io, "In %2d, Out %2d, Weight %+3.5f %6s, Innov %d",
            ng.inId, ng.outId, ng.weight,(ng.enabled? "Enabled":"Disabled"), ng.innovNumber)
    return
end

#     def __cmp__(self, other):
#         return cmp(self.__innov_number, other.__innov_number)

function split(cg::ConnectionGene, node_id::Int64)
    # Splits a connection, creating two new connections and disabling this one """
    cg.enabled = false
    new_conn1 = ConnectionGene(cg.inId, node_id, 1.0, true)
    new_conn2 = ConnectionGene(node_id, cg.outId, cg.weight, true)
    return new_conn1, new_conn2
end

copy(cg::ConnectionGene) = ConnectionGene(cg.inId, cg.outId, cg.weight, cg.enabled, cg.innovNumber)

is_same_innov(self::ConnectionGene, other::ConnectionGene) = self.innovNumber == cg.innovNumber

get_child(self::ConnectionGene, other::ConnectionGene) = randbool? self : other
