
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

g = NEAT.Global()
cg =  NEAT.ConnectionGene(g, 1, 2, .5, true)
cg2 =  NEAT.ConnectionGene(g, 1, 2, .5, true)

# test that two connection with identical in and outs DO have same key and innvotion numbers
@test cg.innovNumber == cg2.innovNumber
@test cg.key == cg2.key

# test that two connection with diffent outs do NOT have same key or innvotion numbers
cg3 =  NEAT.ConnectionGene(g, 1, 3, .5, true)
@test cg3.innovNumber != cg2.innovNumber
@test cg3.key != cg2.key

cg =  NEAT.ConnectionGene(NEAT.Global(), 1, 2, .5, true)
w = cg.weight
NEAT.mutate_weight!(cg,params)
@test cg.weight != w

cg =  NEAT.ConnectionGene(NEAT.Global(), 1, 2, .5, true); w = cg.weight
NEAT.weight_replaced!(cg,params)
@test cg.weight != w

# test that mutate someitme actual does mutate (better test to come)
@test sum(map(x->(NEAT.mutate!(cg, params)== true)?1:0,[1:100])) > 0

g = NEAT.Global()
cg =  NEAT.ConnectionGene(g, 1, 2, .5, true)
@test length(g.innovations) == 1
cg2, cg3 = NEAT.split(g,cg,3)
@test length(g.innovations) == 3
@test cg2.inId == cg.inId
@test cg2.outId != cg.outId
@test cg3.inId != cg.inId
@test cg3.outId == cg.outId

@test NEAT.is_same_innov(cg, cg) == true
@test NEAT.is_same_innov(cg, cg2) == false
@test NEAT.is_same_innov(cg, cg3) == false
@test NEAT.is_same_innov(cg2, cg3) == false

# test for choice between one or other
cg3 = NEAT.get_child(cg,cg2)
@test (cg3==cg || cg3 == cg2) == true
@test (cg3==cg && cg3 == cg2) == false

# different object but with same values
cg2 = NEAT.copy(g,cg)
@test cg2 != cg
@test cg2.inId == cg.inId
@test cg2.outId == cg.outId
@test cg2.weight == cg.weight
@test cg2.enable == cg.enable
@test cg2.innovNumber == cg.innovNumber
