# Recurrent type
g = NEAT.Global(config)
g.cf.feedforward = false
g.cf.nn_activation = :sigm
ch1 = NEAT.create_unconnected(g)
NEAT.add_hidden_nodes!(g,ch1, 2, NEAT.Recurrent())
rec = NEAT.createPhenotype(ch1)
@test length(rec.neurons) == length(ch1.node_genes)
@test length(rec.synapses) == length(ch1.connection_genes)
@test rec.nntype == ch1.node_gene_type
@test rec.numInputs == ch1.inputCnt

@test typeof(NEAT.activate(rec.nntype, rec, [1.,1.,1.])) == Vector{Float64}
@test NEAT.activate(rec.nntype, rec, [1.,1.,1.])[1] > 0. # due to sigmoids

g = NEAT.Global(config)
g.cf.prob_addnode = 0.5
p = NEAT.Population(g)
p.evaluate = (chs)->map(ch->ch.fitness=max(0,randn()*0.3+ch.fitness*0.7), chs)
NEAT.epoch(g, p, 10, false)
rec = NEAT.createPhenotype(p.best_fitness[end])
@test typeof(NEAT.activate(rec.nntype, rec, [1.,1.,1.])) == Vector{Float64}
@test NEAT.activate(rec.nntype, rec, [1.,1.,1.])[1] > 0. # due to sigmoids


# FeedForward type
g = NEAT.Global(config)
g.cf.feedforward = true
ch2 = NEAT.create_unconnected(g)
NEAT.add_hidden_nodes!(g,ch2, 2,NEAT.FeedForward())
ff =  NEAT.createPhenotype(ch2)
@test length(ff.neurons) == length(ch2.node_genes)
@test length(ff.synapses) == length(ch2.connection_genes)
@test ff.nntype == ch2.node_gene_type
@test ff.numInputs == ch2.inputCnt

@test typeof(NEAT.activate(ff.nntype, ff, [1.,1.,1.])) == Vector{Float64}
@test NEAT.activate(ff.nntype, ff, [1.,1.,1.])[1] > 0. # due to sigmoids

g = NEAT.Global(config)
g.cf.prob_addnode = 0.5
p = NEAT.Population(g)
p.evaluate = (chs)->map(ch->ch.fitness=max(0,randn()*0.3+ch.fitness*0.7), chs)
NEAT.epoch(g, p, 10, false)
ff = NEAT.createPhenotype(p.best_fitness[end])
@test typeof(NEAT.activate(ff.nntype, rec, [1.,1.,1.])) == Vector{Float64}
@test NEAT.activate(rec.nntype, rec, [1.,1.,1.])[1] > 0. # due to sigmoids
