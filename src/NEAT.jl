module NEAT

type NeatParams
    inputs::Int64
    outputs::Int64
    function NeatParams(inputs::Int64,outputs::Int64)
       new(inputs,outputs)
    end
end

include("population.jl")
include("genome.jl")
include("species.jl")

# export Input, Output, Hidden


# TO DOs:

# 1. Mutation
# 2. Crossover
# 3. Feed forward
# 4. Fitness evaluation
# 5. Speciation
# 6. Initialization


end # module
