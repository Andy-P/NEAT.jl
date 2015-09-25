module NEAT

import Base.split

include("config.jl")

# track count of various global variables & holds refernce to the config
type Global
    speciesCnt::Int64
    chromosomeCnt::Int64
    nodeCnt::Int64
    innov_number::Int64
    innovations::Dict{(Int64,Int64),Int64}
    cf::Config
    function Global(cf::Config)
        new(0,0,0,0,Dict{(Int64,Int64),Int64}(),cf) # global dictionary
    end
end

include("genome.jl")
include("chromosome.jl")
include("species.jl")
include("population.jl")
include("../networks/nn.jl")

# export Input, Output, Hidden

end # module
