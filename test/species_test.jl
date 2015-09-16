
g = NEAT.Global(config) # reset globals
s1 = NEAT.Species(g, NEAT.Chromosome(g, 1, 2, NEAT.Recurrent(),:ConnectionGene))
@test s1.id == 1
@test length(s1) == 1
@test s1.representant == s1.subpopulation[1]
@test s1.representant.species_id == 1

NEAT.add(s1, NEAT.Chromosome(g, 1, 2, NEAT.Recurrent(),:ConnectionGene))
@test s1.id == 1
@test length(s1) == 2
@test s1.subpopulation[2].species_id == 1

# tests of species id increment && of no parameter sharing
s2 = NEAT.Species(g, NEAT.Chromosome(g, 1, 2, NEAT.Recurrent(),:ConnectionGene))
@test s2.id == 2
@test length(s2) == 1
@test s2.representant == s2.subpopulation[1]
@test s2.representant.species_id == 2


# setup
g = NEAT.Global(config) # reset globals
s3 = NEAT.Species(g, NEAT.Chromosome(g, 1, 2, NEAT.Recurrent(),:ConnectionGene))
NEAT.add(s3, NEAT.Chromosome(g, 1, 2, NEAT.Recurrent(),:ConnectionGene))
NEAT.add(s3, NEAT.Chromosome(g, 1, 2, NEAT.Recurrent(),:ConnectionGene))
s3.subpopulation[1].fitness = 1.
s3.subpopulation[2].fitness = 2.
s3.subpopulation[3].fitness = 3.

# avg fitness & no_improvement_age tests
@test NEAT.average_fitness(s3) == 2.
map(x->NEAT.average_fitness(s3),1:100) # run 100 more times wiht no change
@test s3.no_improvement_age == 100

# tournament selection test
best = NEAT.tournamentSelection(s3)
@test all(ch->ch.id !=1, map(i -> NEAT.tournamentSelection(s3),[1:100]))

# reproduce() tests
g = NEAT.Global(config) # reset globals
s = NEAT.Species(g,NEAT.create_fully_connected(g))
NEAT.add(s,NEAT.create_fully_connected(g))
# NEAT.add(s,NEAT.create_fully_connected(g))
# NEAT.add(s,NEAT.create_fully_connected(g))
spawn_amount = 3
s.spawn_amount = spawn_amount
offspring = NEAT.reproduce(g,s)
@test length(offspring) == spawn_amount
@test offspring[1].id == 3
@test offspring[2].id == 4
@test offspring[3].id == 5

