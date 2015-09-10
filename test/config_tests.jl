using Base.Test

paramsDict = NEAT.loadConfig(joinpath(dirname(@__FILE__),"..","examples","doublePole","dp_config.txt"))
params = NEAT.Config(paramsDict)
@test typeof(params) == NEAT.Config
