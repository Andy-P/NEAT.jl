g = NEAT.Global(config)
g.cg.feedforward = false
p = NEAT.Population(g)
@test all(ch-> ch.node_gene_type == NEAT.Recurrent(), p.population)

g.cg.feedforward = true
p = NEAT.Population(g)
@test all(ch-> ch.node_gene_type == NEAT.FeedForward(), p.population)

g.cg.fully_connected = true
g.cg.feedforward = true
g.cg.hidden_nodes = 0
p = NEAT.Population(g)
@test all(ch-> length(ch.connection_genes) == 3, p.population)

g.cg.fully_connected = true
g.cg.feedforward = true
g.cg.hidden_nodes = 2
p = NEAT.Population(g)
@test all(ch-> length(ch.connection_genes) == 11, p.population)

g.cg.fully_connected = true
g.cg.feedforward = false
g.cg.hidden_nodes = 2
p = NEAT.Population(g)
@test all(ch-> length(ch.connection_genes) == 17, p.population)

g.cg.fully_connected = false
g.cg.feedforward = false
g.cg.hidden_nodes = 2
p = NEAT.Population(g)
@test all(ch-> length(ch.connection_genes) == 14, p.population)

g.cg.fully_connected = false
g.cg.feedforward = true
g.cg.hidden_nodes = 2
p = NEAT.Population(g)
@test all(ch-> length(ch.connection_genes) == 8, p.population)

# remove tests
p = NEAT.Population(g)
map(x->NEAT.remove(p, p.population[1]), 1:5)
@test length(p.population) == g.cg.pop_size - 5

p = NEAT.Population(g)
map(ch->ch.fitness = rand(), p.population)
NEAT.speciate(g, p, true)
NEAT.compute_spawn_levels(g, p)
totalSpawns = sum([s.spawn_amount for s in p.species])
@test length(p.population)-1 <= totalSpawns <= length(p.population)+1
