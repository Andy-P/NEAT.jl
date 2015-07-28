using NEAT
reload("NEAT.jl")
# output requests values to sum from X1, X2,...

# Create 2 input nodes
x0 = NEAT.BiasNode()
x1 = NEAT.InputNode(1.)
x2 = NEAT.InputNode(2.)

# Create 2 hidden nodes
h1 = NEAT.GeneNode(1)
h2 = NEAT.GeneNode(2)

# create connect from the inputs to hidden 1 and wire them up
w1 = NEAT.GeneConnection(x1,0,0,0)
w2 = NEAT.GeneConnection(x2,0,0,1)
push!(h1.xs, w1); w1.out = h1
push!(h1.xs, w2); w2.out = h1

# create connect from the inputs to hidden 2 and wire them up
w3 = NEAT.GeneConnection(x1,0,0,2)
w4 = NEAT.GeneConnection(x2,0,0,3)
push!(h2.xs, w3); w3.out = h2
push!(h2.xs, w4); w4.out = h2

# create connection from each hidden node to each output node
w5 = NEAT.GeneConnection(h1,0,0,3) # h1 ->  o1
w6 = NEAT.GeneConnection(h2,0,0,4) # h2 ->  o1
w7 = NEAT.GeneConnection(h1,0,0,5) # h1 ->  o2
w8 = NEAT.GeneConnection(h2,0,0,6) # h2 ->  o2

o1 = NEAT.GeneNode(5,[w5,w6],0.)
o2 = NEAT.GeneNode(6,[w7,w8],0.)

o1.f()
o2.f()

g = NEAT.Genome(2,2,[o1,o2],[x1,x2],[h1,h2],[w1,w2,w3,w4,w5,w6,w7,w8])
NEAT.addNode(g,2)
