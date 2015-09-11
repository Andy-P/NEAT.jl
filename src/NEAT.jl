module NEAT

import Base.split

# fills roll of class variables
type Global
    nodeCnt::Int64
    innov_number::Int64
    innovations::Dict{(Int64,Int64),Int64}
    function Global()
        new(0,0,Dict{(Int64,Int64),Int64}()) # global dictionary
    end
end
include("config.jl")
include("genome.jl")
# include("population.jl")
# include("species.jl")

# export Input, Output, Hidden


# TO DOs:

# 1. Mutation
# 2. Crossover
# 3. Feed forward
# 4. Fitness evaluation
# 5. Speciation
# 6. Initialization


end # module
