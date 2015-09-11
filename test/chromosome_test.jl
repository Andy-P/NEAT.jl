
ch = NEAT.Chromosome(g, 1, 2,:NodeGene,:ConnectionGene)
@test typeof(ch) == NEAT.Chromosome
@test typeof(ch.id) == Int64
@test typeof(ch.inputCnt) == Int64
@test typeof(ch.outputCnt) == Int64
@test typeof(ch.node_gene_type) == Symbol
@test typeof(ch.conn_gene_type) == Symbol
# @test typeof(ch.connection_genes) == Symbol
# @test typeof(ch.node_genes) == Symbol
@test typeof(ch.fitness) == Function
@test typeof(ch.species_id) == Int64
@test typeof(ch.parent1_id) == Int64
@test typeof(ch.parent2_id) == Int64

g = NEAT.Global(config)
NEAT.incChromeId!(g);NEAT.incChromeId!(g)
@test g.chromosomeCnt == 2

NEAT.mutate_add_node(ch,g)
