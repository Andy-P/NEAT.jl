
type Population
    # Manages all the species
    population::Vector{Chromosome}
    popsize::Int64
    species::Vector{Species}
    species_history::Vector{Array{Int64,1}}
    generation::Int64
    avg_fitness::Vector{Float64}
    best_fitness::Vector{Chromosome}
    evaluate::Function # Evaluates population. Override this method in your experiments
    function Population(g::Global, checkpoint::String="")

        if checkpoint != ""
            # start from a previous point: creates an 'empty'
            # population and point its __dict__ to the previous one
            resume_checkpoint(checkpoint)
        else
            population = Chromosome[]
            popsize = g.cf.pop_size # total population size
            for i in 1:popsize

                ch = g.cf.fully_connected? create_fully_connected(g) : create_minimally_connected(g)
                if g.cf.hidden_nodes > 0
                    ch = ch = g.cf.fully_connected? ch: create_unconnected(g)
                    nType = g.cf.feedforward? FeedForward():Recurrent()
                    add_hidden_nodes!(g, ch, g.cf.hidden_nodes,nType)
                end
                push!(population, ch)
            end

            p = new(population, popsize,
                    Species[],Array{Int64,1}[], # currently living species and species history
                    -1, # generation
                    Float64[], # avg_fitness
                    Chromosome[]); # best_fitness
            p.evaluate = (f::Function) -> f(p.population); # Evaluates population. Override this method in your experiments
            p
        end
    end
end

function Base.show(io::IO, p::Population)
    @printf(io,"Population size: %3d   Total species: %3d", p.popsize, length(p.species))
end

function remove(p::Population, ch::Chromosome)
    # Removes a chromosome from the population
    deleteat!(p.population,findfirst(p.population,ch))
    return
end

function speciate(g::Global, p::Population, report::Bool)
    # Group chromosomes into species by similarity
        # Speciate the population
    for individual in p.population
        found = false
        for s in p.species
            if distance(individual, s.representant, g.cf) < g.cf.compatibility_threshold
                add(s, individual)
                found = true
                break
            end
        end

        if !found push!(p.species, Species(g, individual)) end
    end

    # eliminate empty species
    keep = map(s->length(s)==0?false:true,p.species)
    if report
        for i = 1:length(keep)
            if !keep[i] println("Removing species $(p.species[i].id) for being empty") end
        end
    end
    p.species = p.species[keep]
    set_compatibility_threshold(g, p)

end

function set_compatibility_threshold(g::Global, p::Population)
    # controls compatibility threshold
    if length(p.species) > g.cf.species_size
        g.cf.compatibility_threshold += g.cf.compatibility_change
    elseif length(p.species) < g.cf.species_size
        if g.cf.compatibility_threshold > g.cf.compatibility_change
            g.cf.compatibility_threshold -= g.cf.compatibility_change
        else
            println("Compatibility threshold cannot be changed (minimum value has been reached)")
        end
    end
end

# Returns the average raw fitness of population
average_fitness(p::Population) = mean([p.population[i].fitness::Float64 for i=1:length(p.population)])

stdeviation(p::Population) = std([p.population[i].fitness::Float64 for i=1:length(p.population)])

function compute_spawn_levels(g::Global, p::Population)
    #  Compute each species' spawn amount (Stanley, p. 40)

    # 1. Boost if young and penalize if old
    # TODO: does it really increase the overall performance?
    species_stats = zeros(length(p.species))
    for i = 1:length(p.species)
        s = p.species[i]
        species_stats[i] = s.age < g.cf.youth_threshold? average_fitness(s) * g.cf.youth_boost:
            s.age > g.cf.old_threshold? average_fitness(s) * g.cf.youth_boost : average_fitness(s)
        if s.age < g.cf.youth_threshold
            species_stats[i] = average_fitness(s) * g.cf.youth_boost
        elseif s.age > g.cf.old_threshold
           species_stats[i] = average_fitness(s) * g.cf.old_penalty
        else
            species_stats[i] = average_fitness(s)
        end
    end

    # 2. Share fitness (only usefull for computing spawn amounts)
    # More info: http://tech.groups.yahoo.com/group/neat/message/2203
    # Sharing the fitness is only meaningful here
    # we don't really have to change each individual's raw fitness
    total_average = sum(species_stats)

     # 3. Compute spawn
    for i= 1:length(p.species)
        s = p.species[i]
        s.spawn_amount = int(round((species_stats[i] * p.popsize / total_average)))
    end
end

function tournamentSelection(p::Population, k=2)
    # Tournament selection with size k (default k=2).
    # randomly select k competitors
    chs = p.population[randperm(length(p.population))[1:k]]
    best = chs[1]
    for ch in chs # choose best among randomly selected
        best = ch.fitness > best.fitness? ch :best
    end
    return best
end

function log_species(p::Population)
    # Logging species data for visualizing speciation
    specById = sort(p.species, by=s->s.id)
    spec_size = zeros(Int64,specById[end].id+1)
    map(s->spec_size[s.id]=length(s) , specById)
    push!(p.species_history,spec_size)
end

function population_diversity(p::Population)
    # Calculates the diversity of population: total average weights,
    # number of connections, nodes

    num_nodes = 0
    num_conns = 0
    avg_weights = 0.0

    for ch in p.population
        num_nodes += length(ch.node_genes)
        num_conns += length(ch.conn_genes)
        for cg in ch.conn_genes
            avg_weights += cg.weight
        end
    end

    total = length(p.population)
    return num_nodes/total, num_conns/total, avg_weights/total
end

function epoch(g::Global, p::Population, n::Int64, report::Bool=true, save_best::Bool=false,
               checkpoint_interval::Int64=15, checkpoint_generation=0)
    #= Runs NEAT's genetic algorithm for n epochs.

        Keyword arguments:
        report -- show stats at each epoch (default true)
        save_best -- save the best chromosome from each epoch (default False)
        checkpoint_interval -- time in minutes between saving checkpoints (default 15 minutes)
        checkpoint_generation -- time in generations between saving checkpoints
            (default 0 -- option disabled)
    =#
    t0 = time() # for saving checkpoints

    for gen in 0:n
        p.generation += 1

        if report println(" ****** Running generation $(p.generation) ******") end

        # Evaluate individuals
        p.evaluate(p.population)
        # Speciates the population
        speciate(g, p, report)

        # Current population's average fitness
        push!(p.avg_fitness, average_fitness(p))

        # Current generation's best chromosome
        bestfit, bestidx = findmax(map(ch-> ch.fitness, p.population))
        best = p.population[bestidx]
        push!(p.best_fitness, best)

        # Which species has the best chromosome?
        for s in p.species
            s.hasBest = false
            if best.species_id == s.id
                s.hasBest = true
            end
        end

#         # saves the best chromo from the current generation
#         if save_best
#             file = open('best_chromo_'+str(self.__generation),'w')
#             pickle.dump(best, file)
#             file.close()
#               end

        #-----------------------------------------
        # Prints chromosome's parents id:  {dad_id, mon_id} -> child_id
#         map(ch-> @printf("{%3d; %3d} -> %3d   Nodes %3d   Connections %3d\n",
#                          ch.parent1_id, ch.parent2_id, ch.id, size(ch)[1], size(ch)[2]), p.population)
        #-----------------------------------------

        # Remove stagnated species and its members (except if it has the best chromosome)
        speciesToKeep = trues(length(p.species))
        deletedSpeciesIds = Int64[]
        for i in 1:length(p.species)
            if p.species[i].no_improvement_age > g.cf.max_stagnation
                if !p.species[i].hasBest || p.species[i].no_improvement_age > 2 * g.cf.max_stagnation
                    if report @printf("\n   Species %2d age %2s (with %2d individuals) is stagnated: removing it",
                                      p.species[i].id, p.species[i].age, length(p.species[i])) end
                    speciesToKeep[i] = false
                    push!(deletedSpeciesIds,p.species[i].id)
                end
            end
        end
        p.species = p.species[speciesToKeep] # prune unwanted species

        # remove species' chromosomes from population
        chromosToKeep = trues(length(p.population))
        for i in 1:length(p.population)
            if findfirst(deletedSpeciesIds,p.population[i].species_id) != 0 chromosToKeep[i] = false end
        end
        p.population = p.population[chromosToKeep] # prune unwanted chromosomes

        # Compute spawn levels for each remaining species
        compute_spawn_levels(g, p)

        # Removing species with spawn amount = 0
        speciesToKeep = trues(length(p.species))
        deletedSpeciesIds = Int64[]
        for i in 1:length(p.species)

            # This rarely happens
            if p.species[i].spawn_amount == 0
                if report @printf("\n   Species %2d age %2s removed: produced no offspring",p.species[i].id, p.species[i].age) end
                speciesToKeep[i] = false
                push!(deletedSpeciesIds,p.species[i].id)
            end
        end
        p.species = p.species[speciesToKeep] # prune unwanted species

        # remove species' chromosomes from population
        chromosToKeep = trues(length(p.population))
        for i in 1:length(p.population)
            if findfirst(deletedSpeciesIds,p.population[i].species_id) != 0 chromosToKeep[i] = false end
        end
        p.population = p.population[chromosToKeep] # prune unwanted chromosomes

        # Logging speciation stats
        log_species(p)

        if report
            @printf("\nPopulation's average fitness: %3.5f stdev: %3.5f", p.avg_fitness[end], stdeviation(p))
            @printf("\nBest fitness: %2.12s - size: %s - species %s - id %s", best.fitness, size(best), best.species_id, best.id)

            # print some "debugging" information
            @printf("\nSpecies length: %d totalizing %d individuals", length(p.species), sum([length(s) for s in p.species]))
            @printf("\nSpecies ID       : %s",   [s.id for s in p.species])
            @printf("\nEach species size: %s",   [length(s) for s in p.species])
            @printf("\nAmount to spawn  : %s",   [s.spawn_amount for s in p.species])
            @printf("\nSpecies age      : %s",   [s.age for s in p.species])
            @printf("\nSpecies no improv: %s\n", [s.no_improvement_age for s in p.species]) # species no improvement age

            for s in p.species println(s) end
        end

        # Stops the simulation
        if best.fitness > g.cf.max_fitness_threshold
            @printf("Best individual found in epoch %s - complexity: %s\n", p.generation, size(best))
            break
        end

        # -------------------------- Producing new offspring -------------------------- #
        new_population = Chromosome[] # next generation's population

        # Spawning new population
        for s in p.species new_population = vcat(new_population,reproduce(g, s)) end

        # ----------------------------------------------#
        # Controls target population under or overflow  #
        # ----------------------------------------------#
        fill = p.popsize - length(new_population)
        if fill < 0 # overflow
            if report println("\n   Removing $(abs(fill)) excess individual(s) from the new population") end
            # TODO: This is dangerous? I can't remove a species' representant!
            new_population = new_population[1:end+fill] # Removing the last added members
        end

        if fill > 0 # underflow
            if report println("\n   Producing $fill more individual(s) to fill up the new population") end

            # TODO:
            # what about producing new individuals instead of reproducing?
            # increasing diversity from time to time might help
            while fill > 0
                # Selects a random chromosome from population
                parent1 = p.population[rand(1:length(p.population))]
                # Search for a mate within the same species
                found = false
                for parent2 in p.population
                    if parent2.species_id == parent1.species_id && parent2.id != parent1.id
                        child = mutate(crossover(g, parent1, parent2),g)
                        push!(new_population,child)
                        found = true
                        break
                    end
                end
                if !found
                    push!(new_population, mutate(deepcopy(parent1),g)) # will irreversibly mutate parent. ok?
                end # If no mate was found, just mutate it

                fill -= 1
            end
        end

        @assert p.popsize == length(new_population) # Different population sizes!

        # Updates current population
        p.population = new_population

        if time() > t0 + 60 * checkpoint_interval
            create_checkpoint(p,report)
            t0 = time() # updates the counter
        elseif  checkpoint_generation != 0 && p.generation % checkpoint_generation == 0
            create_checkpoint(p,report)
        end
    end
end

function resume_checkpoint(checkpoint)
    # Resumes the simulation from a previous saved point.
    # try:
    #     #file = open(checkpoint)
    #     file = gzip.open(checkpoint)
    # except IOError:
    #     raise
    # print 'Resuming from a previous point: %s' %checkpoint
    # # when unpickling __init__ is not called again
    # previous_pop = pickle.load(file)
    # self.__dict__ = previous_pop.__dict__

    # print 'Loading random state'
    # rstate = pickle.load(file)
    # random.setstate(rstate)
    # #random.jumpahead(1)
    # file.close()
end

function create_checkpoint(p::Population, report)
        # Saves the current simulation state.
        # from time import strftime
        # get current time
        # date = strftime("%Y_%m_%d_%Hh%Mm%Ss")
        # if report print 'Creating checkpoint file at generation: %d' %self.__generation end

        # # dumps 'self'
        # #file = open('checkpoint_'+str(self.__generation), 'w')
        # file = gzip.open('checkpoint_'+str(self.__generation), 'w', compresslevel = 5)
        # # dumps the population
        # pickle.dump(self, file, protocol=2)
        # # dumps the current random state
        # pickle.dump(random.getstate(), file, protocol=2)
        # file.close()
end

# if __name__ ==  '__main__' :

#     # sample fitness function
#     def eval_fitness(population):
#         for individual in population:
#             individual.fitness = 1.0

#     # set fitness function
#     Population.evaluate = eval_fitness

#     # creates the population
#     pop = Population()
#     # runs the simulation for 250 epochs
#     pop.epoch(250)

