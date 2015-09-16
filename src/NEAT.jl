module NEAT

import Base.split

include("config.jl")

# fills roll of class variables
type Global
    speciesCnt::Int64
    chromosomeCnt::Int64
    nodeCnt::Int64
    innov_number::Int64
    innovations::Dict{(Int64,Int64),Int64}
    cg::Config
    function Global(cg::Config)
        new(0,0,0,0,Dict{(Int64,Int64),Int64}(),cg) # global dictionary
    end
end

include("genome.jl")
include("chromosome.jl")

# include("population.jl")
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
