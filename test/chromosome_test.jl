
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

nodeCnt = length(ch.node_genes)
ng, cg = NEAT.mutate_add_node!(ch,g)
@test cg.enable == false
@test length(ch.connection_genes) == g.cg.input_nodes * g.cg.output_nodes + 2
@test length(ch.node_genes) == nodeCnt + 1

connectCnt = length(ch.connection_genes)
NEAT.mutate_add_connection!(ch, g)
@test length(ch.connection_genes) ==connectCnt + 1

@test NEAT.size(ch) == (1,5)
