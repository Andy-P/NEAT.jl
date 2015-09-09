using NEAT
using Base.Test

@osx_only params = NEAT.load(joinpath(dirname(@__FILE__),"../examples//doublePole","dp_config.txt"))
@windows_only params = NEAT.load(joinpath(dirname(@__FILE__),"../examples//doublePole","dp_config.txt"))

params = NEAT.Config(params)
