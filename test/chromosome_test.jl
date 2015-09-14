
ch = NEAT.Chromosome(g, 1, 2,:NodeGene,:ConnectionGene)
@test typeof(ch) == NEAT.Chromosome
@test typeof(ch.id) == Int64
@test typeof(ch.inputCnt) == Int64
@test typeof(ch.outputCnt) == Int64
@test typeof(ch.node_gene_type) == Symbol
@test typeof(ch.conn_gene_type) == Symbol
# @test typeof(ch.connection_genes) == Symbol
# @test typeof(ch.node_genes) == Symbol
@test typeof(ch.fitness) == Float64
@test typeof(ch.species_id) == Int64
@test typeof(ch.parent1_id) == Int64
@test typeof(ch.parent2_id) == Int64

g = NEAT.Global(config)
NEAT.incChromeId!(g);NEAT.incChromeId!(g)
@test g.chromosomeCnt == 2

ch = NEAT.create_minimally_connected(g)
@test length(ch.node_genes) == g.cg.input_nodes + g.cg.output_nodes
@test length(ch.connection_genes) == g.cg.output_nodes

ch = NEAT.create_fully_connected(g)
@test length(ch.node_genes) == g.cg.input_nodes + g.cg.output_nodes
@test length(ch.connection_genes) == g.cg.input_nodes * g.cg.output_nodes
@test NEAT.maxInnov(ch.connection_genes).innovNumber == 3

nodeCnt = length(ch.node_genes)
ng, cg = NEAT.mutate_add_node!(ch,g)
@test cg.enable == false
@test length(ch.connection_genes) == g.cg.input_nodes * g.cg.output_nodes + 2
@test length(ch.node_genes) == nodeCnt + 1

connectCnt = length(ch.connection_genes)
NEAT.mutate_add_connection!(ch, g)
@test length(ch.connection_genes) ==connectCnt + 1

@test NEAT.size(ch) == (1,5)

ch1 = NEAT.create_fully_connected(g)
ng, cg = NEAT.mutate_add_node!(ch1,g)
ng, cg = NEAT.mutate_add_node!(ch1,g)
ng, cg = NEAT.mutate_add_node!(ch1,g)
@test length(ch1.node_genes) == 7
@test length(ch1.connection_genes) == 9

ch2 = NEAT.create_fully_connected(g)
ng, cg = NEAT.mutate_add_node!(ch2,g)
ng, cg = NEAT.mutate_add_node!(ch2,g)
@test length(ch2.node_genes) == 6
@test length(ch2.connection_genes) == 7

@test NEAT.distance(g,NEAT.
    create_fully_connected(g),NEAT.create_fully_connected(g)) > 0

child = NEAT.Chromosome(g, ch1.id, ch2.id, :NodeGene,:ConnectionGene)
NEAT.inherit_genes(g, child, ch1, ch2)
@test length(child.node_genes) == 7 # test that it is larger of two
@test length(child.connection_genes) == 9 # test that it is larger of two

child = NEAT.crossover(g, ch1, ch2)
@test length(child.node_genes) == 7 # test that it is larger of two
@test length(child.connection_genes) == 9 # test that it is larger of two

child
