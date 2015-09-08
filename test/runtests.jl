using NEAT
using Base.Test

# create 2 input nodes

# create 3 hidden nodes

# create 2 output nodes

# connect:
# I1 -> H1
# I2 -> H2
# H2 -> H3
# H2 -> H1
# H1 -> H3
 # H2 -> O1
# H1 -> O2
# H2 -> O2

# test forward

# test recursive
# change connections to creat loop
#  H2 -> H1 become  H1 -> H2
#  H1 -> H3 become  H3 -> H1


# if __name__ == "__main__":
#     # Example
#     #from neat import visualize

#     nn = FeedForward([2,10,3], use_bias=False, activation_type = 'exp')
#     ##visualize.draw_ff(nn)
#     print 'Serial activation method: '
#     for t in range(3):
#         print nn.sactivate([1,1])

#     #print 'Parallel activation method: '
#     #for t in range(3):
#         #print nn.pactivate([1,1])

#     # defining a neural network manually
#     #neurons = [Neuron('INPUT', 1), Neuron('HIDDEN', 2), Neuron('OUTPUT', 3)]
#     #connections = [(1, 2, 0.5), (1, 3, 0.5), (2, 3, 0.5)]

#     #net = Network(neurons, connections) # constructs the neural network
#     #visualize.draw_ff(net)
#     #print net.pactivate([0.04]) # parallel activation method
#     #print net # print how many neurons and synapses our network has
