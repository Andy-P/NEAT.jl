
ng = NEAT.NodeGene(1, :INPUT)
@test typeof(ng.id) == Int64
@test typeof(ng.ntype) == Symbol
@test typeof(ng.bias) == Float64
@test typeof(ng.response) == Float64
@test typeof(ng.activation) == Symbol


ng = NEAT.NodeGene(1, :INPUT, 0.,4.9, :sigm)
@test ng.ntype == :INPUT
@test ng.ntype != :HIDDEN
@test ng.ntype != :OUTPUT

ng = NEAT.NodeGene(1, :HIDDEN, 0.,4.9, :sigm)
@test ng.ntype != :INPUT
@test ng.ntype == :HIDDEN
@test ng.ntype != :OUTPUT

ng = NEAT.NodeGene(1, :OUTPUT, 0.,4.9, :sigm)
@test ng.ntype != :INPUT
@test ng.ntype != :HIDDEN
@test ng.ntype == :OUTPUT

child = NEAT.get_child(ng,ng)
@test child.id         == ng.id
@test child.ntype      == ng.ntype
@test child.bias       == ng.bias
@test child.response   == ng.response
@test child.activation == ng.activation

ng = NEAT.NodeGene(1, :HIDDEN, 0., 1., :sigm)
NEAT.mutate_bias!(ng, params)
@test ng.bias != 0

ng = NEAT.NodeGene(1, :HIDDEN, 0., 1., :sigm)
NEAT.mutate_response!(ng,params)
@test ng.response != 1

ng = NEAT.NodeGene(1, :HIDDEN, 0., 1., :sigm)
NEAT.mutate!(ng, params) # make sure it runs :()

ctng = NEAT.CTNodeGene(1, :HIDDEN)
@test typeof(ctng.id) == Int64
@test typeof(ctng.ntype) == Symbol
@test typeof(ctng.bias) == Float64
@test typeof(ctng.response) == Float64
@test typeof(ctng.activation) == Symbol
@test typeof(ctng.timeConstant) == Float64

@test (ctng.id) == 1
@test (ctng.ntype) == :HIDDEN
@test (ctng.bias) == 0.
@test (ctng.response) == 4.924273
@test (ctng.activation) == :sigm
@test (ctng.timeConstant) == 1.

ng = NEAT.CTNodeGene(1, :HIDDEN)
NEAT.mutate!(ng, params)
@test typeof(ng) == NEAT.CTNodeGene
ng2 = NEAT.copy(ng)
@test typeof(ng2) == NEAT.CTNodeGene
@test ng2 != ng # not same object

child = NEAT.get_child(ng, ng2)
@test child != ng # not same object
@test child.id         == ng.id
@test child.ntype      == ng.ntype
@test child.bias       == ng.bias
@test child.response   == ng.response
@test child.activation == ng.activation

cg =  NEAT.ConnectionGene(1, 2, .5, true)
global_innov_number = NEAT.get_new_innov_number(global_innov_number)

