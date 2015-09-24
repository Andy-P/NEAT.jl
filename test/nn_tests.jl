reload("NEAT")
include("config_tests.jl")

# NEAT.Network(3)

# Recurrent type
g = NEAT.Global(config)
g.cg.feedforward = false
ch1 = NEAT.create_unconnected(g)
NEAT.add_hidden_nodes!(g,ch1, 2,NEAT.Recurrent())
rec = NEAT.createPhenotype(ch1)
NEAT.activate(rec.nntype, rec, [1.,1.,1.])

rec.synapses[1]
# FeedForward type
g = NEAT.Global(config)
g.cg.feedforward = true
ch1 = NEAT.create_unconnected(g)
NEAT.add_hidden_nodes!(g,ch1, 2,NEAT.FeedForward())
ff =  NEAT.createPhenotype(ch1)
NEAT.activate(ff.nntype, ff, [1.,1.,1.])
ff.neurons
ff.synapses[1]

