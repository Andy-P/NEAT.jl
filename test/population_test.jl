g = NEAT.Global(config)
g.cf.feedforward = false
p = NEAT.Population(g)
@test all(ch-> ch.node_gene_type == NEAT.Recurrent(), p.population)

g.cf.feedforward = true
p = NEAT.Population(g)
@test all(ch-> ch.node_gene_type == NEAT.FeedForward(), p.population)

g.cf.fully_connected = true
g.cf.feedforward = true
g.cf.hidden_nodes = 0
p = NEAT.Population(g)
@test all(ch-> length(ch.connection_genes) == 3, p.population)

g.cf.fully_connected = true
g.cf.feedforward = true
g.cf.hidden_nodes = 2
p = NEAT.Population(g)
@test all(ch-> length(ch.connection_genes) == 11, p.population)

g.cf.fully_connected = true
g.cf.feedforward = false
g.cf.hidden_nodes = 2
p = NEAT.Population(g)
@test all(ch-> length(ch.connection_genes) == 17, p.population)

g.cf.fully_connected = false
g.cf.feedforward = false
g.cf.hidden_nodes = 2
p = NEAT.Population(g)
@test all(ch-> length(ch.connection_genes) == 14, p.population)

g.cf.fully_connected = false
g.cf.feedforward = true
g.cf.hidden_nodes = 2
p = NEAT.Population(g)
@test all(ch-> length(ch.connection_genes) == 8, p.population)

# remove tests
p = NEAT.Population(g)
map(x->NEAT.remove(p, p.population[1]), 1:5)
@test length(p.population) == g.cf.pop_size - 5

g = NEAT.Global(config)
p = NEAT.Population(g)
map(ch->ch.fitness = rand(), p.population)
NEAT.speciate(g, p, true)
NEAT.compute_spawn_levels(g, p)
totalSpawns = sum([s.spawn_amount for s in p.species])
@test length(p.population)-1 <= totalSpawns <= length(p.population)+1

g = NEAT.Global(config)
g.cf.prob_addnode = 0.5
p = NEAT.Population(g)
p.evaluate = (chs)->map(ch->ch.fitness=max(0,randn()*0.3+ch.fitness*0.7), chs)
@time NEAT.epoch(g, p, 1, false)
