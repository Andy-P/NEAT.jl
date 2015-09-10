
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
