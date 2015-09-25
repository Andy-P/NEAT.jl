type Config

    # phenotype config
    input_nodes::Int64
    output_nodes::Int64
    hidden_nodes::Int64
    fully_connected::Bool
    max_weight::Float64
    min_weight::Float64
    max_weight_tau::Float64
    min_weight_tau::Float64
    feedforward::Bool
    nn_activation::Symbol
    weight_stdev::Float64

    # GA config
    pop_size::Int64
    max_fitness_threshold::Float64
    prob_addconn::Float64
    prob_addnode::Float64
    prob_mutatebias::Float64
    bias_mutation_power::Float64
    prob_mutate_weight::Float64 # dynamic mutation rate (future release)
    weight_mutation_power::Float64
    prob_togglelink::Float64
    elitism::Bool

    # genotype compatibility
    compatibility_threshold::Float64
    compatibility_change::Float64
    excess_coeficient::Float64
    disjoint_coeficient::Float64
    weight_coeficient::Float64

    # species
    species_size::Int64
    survival_threshold::Float64 # only the best 20% for each species is allowed to mate
    old_threshold::Int64
    youth_threshold::Int64
    old_penalty::Float64    # always in (0,1)
    youth_boost::Float64    # always in (1,2)
    max_stagnation::Int64

    function Config(params::Dict{String,String})

        new(
            # phenotype
            int(params["input_nodes"]),
            int(params["output_nodes"]),
            int(params["hidden_nodes"]),
            bool(int(params["fully_connected"])),
            float(params["max_weight"]),
            float(params["min_weight"]),
            float(params["max_weight_tau"]),
            float(params["min_weight_tau"]),
            bool(int(params["feedforward"])),
            symbol(params["nn_activation"]),
            float(params["weight_stdev"]),

            # GA
            int(params["pop_size"]),
            float(params["max_fitness_threshold"]),
            float(params["prob_addconn"]),
            float(params["prob_addnode"]),
            float(params["prob_mutatebias"]),
            float(params["bias_mutation_power"]),
            float(params["prob_mutate_weight"]),
            float(params["weight_mutation_power"]),
            float(params["prob_togglelink"]),
            bool(int(params["elitism"])),

            # genotype compatibility
            float(params["compatibility_threshold"]),
            float(params["compatibility_change"]),
            float(params["excess_coeficient"]),
            float(params["disjoint_coeficient"]),
            float(params["weight_coeficient"]),

            # species
            int(params["species_size"]),
            float(params["survival_threshold"]),
            int(params["old_threshold"]),
            int(params["youth_threshold"]),
            float(params["old_penalty"]),
            float(params["youth_boost"]),
            int(params["max_stagnation"])
         )
    end
end

function loadConfig(file::String)
    str = readall(file)

    str = replace(str, r"\r(\n)?", '\n')
    ls = split(str, "\n")

    ls = filter(l->length(l)>0 && l[1] != '#', ls)
    lsMap = map(x->split(x,'='),ls)
    params = Dict{String,String}()
    for i = 1:length(lsMap)
#         println((i,lsMap[i][1],lsMap[i][2]))
        params[rstrip(lsMap[i][1])] = lstrip(lsMap[i][2])
    end

    return params
end
