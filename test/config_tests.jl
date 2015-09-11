using Base.Test

paramsDict = NEAT.loadConfig(joinpath(dirname(@__FILE__),"..","examples","doublePole","dp_config.txt"))
config = NEAT.Config(paramsDict)
g = NEAT.Global(config)
@test typeof(config) == NEAT.Config
