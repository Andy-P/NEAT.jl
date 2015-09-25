type Species
    # A subpopulation containing similar individiduals
    id::Int64                         # species's id
    age::Int64                        # species's age
    subpopulation::Vector{Chromosome} # species's individuals
    hasBest::Bool                     # Does this species has the best individual of the population?
    spawn_amount::Int64
    no_improvement_age::Int64         # the age species has shown no improvements on average
    last_avg_fitness::Float64
    representant::Chromosome
    function Species(g::Global, first_individual::Chromosome, previous_id=0)
        # A species requires at least one individual to come to existence
        id  = previous_id == 0? get_new_id(g):previous_id
        first_individual.species_id = id
        new(id,
            0,     # species's age
            [first_individual],    # species' individuals
            false, # hasBest: Does this species has the best individual of the population?
            0,     # spawn_amount
            0,     # the age species has shown no improvements on average
            0.,    # last_avg_fitness
            first_individual)
    end
end

function get_new_id(g::Global)
    g.speciesCnt += 1
    return g.speciesCnt
end

function add(s::Species, individual::Chromosome)
    individual.species_id = s.id
    push!(s.subpopulation, individual)
    # choose a new random representant for the species
    s.representant = s.subpopulation[rand(1:length(s.subpopulation))]
end

# Returns the total number of individuals in this species
Base.length(s::Species) = length(s.subpopulation)

function Base.show(io::IO, s::Species)
    str = @sprintf("   Species %2d   size: %3d   age: %3d   spawn: %3d  ",
                    s.id, length(s.subpopulation), s.age, s.spawn_amount)
    str = @sprintf("%1s  No improvement: %3d   avg. fitness: %1.8f",
                    str,s.no_improvement_age, s.last_avg_fitness)
    @printf(io,"%6s", str)
end

function tournamentSelection(s::Species, k=2)
    # Tournament selection with size k (default k=2).
    # randomly select k competitors
    selected = randperm(length(s.subpopulation))[1:k]
    chs = s.subpopulation[selected]
    best = chs[1]
    for ch in chs # choose best among randomly selected
        best = ch.fitness > best.fitness? ch :best
    end
    return best
end

function average_fitness(s::Species)
    # Returns the raw average fitness for this species
    @assert length(s) > 0
    sum = 0.0
    for ch in s.subpopulation
        sum += ch.fitness
    end
    current = sum / length(s)

    # if no_improvement_age > threshold, species will be removed
    if current > s.last_avg_fitness
        s.last_avg_fitness = current
        s.no_improvement_age = 0
    else
        s.no_improvement_age += 1
    end
    return current
end

function reproduce(g::Global,s::Species)
    # Returns a list of 'spawn_amount' new individuals

    offspring = Chromosome[] # new offspring for this species
    s.age += 1  # increment species age

#     @printf("Reproducing species %d with %d members\n", s.id, length(s))

    # this condition is useless since no species with spawn_amount < 0 will
    # reach this point - at least it shouldn't happen.
    @assert s.spawn_amount > 0

    sort!(s.subpopulation, by= ch-> ch.fitness, rev=true)

    if g.cf.elitism
        # TODO: Wouldn't it be better if we set elitism=2,3,4...
        # depending on the size of each species?
        push!(offspring, s.subpopulation[1])
        s.spawn_amount -= 1
    end

    survivors = ifloor(length(s) * g.cf.survival_threshold) # keep a % of the best individuals
#     println("survivors = $survivors $(length(s))")
    s.subpopulation = survivors > 0? s.subpopulation[1:survivors]: [s.subpopulation[1]]

    while(s.spawn_amount > 0)

        s.spawn_amount -= 1

        if length(s) > 1
            # Selects two parents from the remaining species and produces a single individual
            # Stanley selects at random, here we use tournament selection (although it is not
            # clear if has any advantages)
            parent1 = tournamentSelection(s)
            parent2 = tournamentSelection(s)

            @assert parent1.species_id == parent2.species_id
            child = crossover(g, parent1, parent2)
            push!(offspring, mutate(child,g))
        else
            # mutate only
            parent1 = s.subpopulation[1]
            # TODO: temporary hack - the child needs a new id (not the father's)
            child = crossover(g, parent1, parent1)
            push!(offspring, child)
        end
    end

    # reset species (new members will be added again when speciating)
    s.subpopulation = []

    # select a new random representant member
    s.representant = offspring[rand(1:length(offspring))]

    return offspring
end
