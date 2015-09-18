
type Population
    # Manages all the species
    population::Vector{Chromosome}
    popsize::Int64
    species::Vector{Species}
    species_history
    generation::Int64
    avg_fitness::Vector{Float64}
    best_fitness::Vector{Float64}
    evaluate::Function # Evaluates population. Override this method in your experiments
    function Population(g::Global, checkpoint::String="")

        if checkpoint != ""
            # start from a previous point: creates an 'empty'
            # population and point its __dict__ to the previous one
            resume_checkpoint(checkpoint)
        else
            population = Chromosome[]
            popsize = g.cg.pop_size # total population size
            for i in 1:popsize

                ch = g.cg.fully_connected? create_fully_connected(g) : create_minimally_connected(g)
                if g.cg.hidden_nodes > 0
                    ch = ch = g.cg.fully_connected? ch: create_unconnected(g)
                    nType = g.cg.feedforward? FeedForward():Recurrent()
                    add_hidden_nodes!(g, ch, g.cg.hidden_nodes,nType)
                end
                push!(population, ch)
            end

            new(population, popsize,
                Species[],[], # currently living species and species history
                -1, # generation
                Float64[], # avg_fitness
                Float64[], # best_fitness
                x->0) # Evaluates population. Override this method in your experiments
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
            if distance(g, individual, s.representant) < g.cg.compatibility_threshold
                add(s, individual)
                found = true
                break
            end
        end

        if !found push!(p.species, Species(g, individual)) end
    end


    # python technical note:
    # we need a "working copy" list when removing elements while looping
    # otherwise we might end up having sync issues
    for s in p.species
        # this happens when no chromosomes are compatible with the species
        if length(s) == 0
            if report println("Removing species $(s.id) for being empty") end
            # remove empty species
            deleteat!(p.species,findfirst(p.species,s))
        end
    end

    set_compatibility_threshold(g, p)

end

function set_compatibility_threshold(g::Global, p::Population)
    # ntrols compatibility threshold
    if length(p.species) > g.cg.species_size
        g.cg.compatibility_threshold += g.cg..compatibility_change
    elseif length(p.species) < g.cg.species_size
        if g.cg.compatibility_threshold > g.cg.compatibility_change
            g.cg.compatibility_threshold -= g.cg.compatibility_change
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
#         species_stats[i] = s.age < g.cg.youth_threshold? average_fitness(s) * g.cg.youth_boost:
#             s.age > g.cg.old_threshold? average_fitness(s) * g.cg.youth_boost : average_fitness(s)
        if s.age < g.cg.youth_threshold
            species_stats[i] = average_fitness(s) * g.cg.youth_boost
        elseif s.age > g.cg.old_threshold
           species_stats[i] = average_fitness(s) * g.cg.old_penalty
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
        s.spawn_amount = int(round((species_stats[i]*p.popsize/total_average)))
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
    higher = max([s.id for s in p.species])
    temp = []
    for i in 1:higher+1
        found_species = false
        for s in self.__species
            if i == s.id
                temp.append(len(s))
                found_specie = true
                break
            end
        end

        if !found_species
            temp.append(0)
    p.species_log.append(temp)
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

    for gen in 1:n
        p.generation += 1

        if report println(" ****** Running generation $(p.generation) ******") end

        # Evaluate individuals
#         evaluate(p)
        # Speciates the population
#         speciate(p,report)

        # Current generation's best chromosome
        push!(p.best_fitness, maximum(map(ch-> ch.fitness, p.population)))

        # Current population's average fitness
        push!(p.avg_fitness, average_fitness(p.population))

        # Print some statistics
        best = p.best_fitness[end]

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

#         # Stops the simulation
#         if best.fitness > Config.max_fitness_threshold:
#             print '\nBest individual found in epoch %s - complexity: %s' %(self.__generation, best.size())
#             break

        #-----------------------------------------
        # Prints chromosome's parents id:  {dad_id, mon_id} -> child_id
        map(ch-> @printf("{%3d; %3d} -> %3d",ch.parent1_id, ch.parent2_id, ch.id),p.population)
        #-----------------------------------------


        # Remove stagnated species and its members (except if it has the best chromosome)
        for s in p.species
            if s.no_improvement_age > g.cg.max_stagnation
                if !s.hasBest || s.no_improvement_age > 2*Config.max_stagnation
                    if report @printf("\n   Species %2d age %2s (with %2d individuals) is stagnated: removing it",
                                      s.id, s.age, length(s)) end
                    # removing species
                    deleteat!(p.species,findfirst(p.species,s))
                    # removing all the species' members
                    #TODO: can be optimized!
                    for ch in p.population
                        if ch.species_id == ch.id deleteat!(p.population,findfirst(p.population,ch)) end
                    end
                end
            end
        end

        # Compute spawn levels for each remaining species
        compute_spawn_levels(g, p)

        # Removing species with spawn amount = 0
        for s in p.species
            # This rarely happens
            if s.spawn_amount == 0
                if report @printf("\n   Species %2d age %2s removed: produced no offspring",s.id, s,age) end
                # removing species
                deleteat!(p.species,findfirst(p.species,s))
                # removing all the species' members
                #TODO: can be optimized!
                for ch in p.population
                    if ch.species_id == ch.id deleteat!(p.population,findfirst(p.population,ch)) end
                end
            end

        # Logging speciation stats
        p.log_species()

#         if report
#             #print 'Poluation size: %d \t Divirsity: %s' %(len(self), self.__population_diversity())
#             print 'Population\'s average fitness: %3.5f stdev: %3.5f' %(self.__avg_fitness[-1], self.stdeviation())
#             print 'Best fitness: %2.12s - size: %s - species %s - id %s' \
#                 %(best.fitness, best.size(), best.species_id, best.id)
#         end

#             # print some "debugging" information
#             print 'Species length: %d totalizing %d individuals' \
#                     %(len(self.__species), sum([len(s) for s in self.__species]))
#             print 'Species ID       : %s' % [s.id for s in self.__species]
#             print 'Each species size: %s' % [len(s) for s in self.__species]
#             print 'Amount to spawn  : %s' % [s.spawn_amount for s in self.__species]
#             print 'Species age      : %s' % [s.age for s in self.__species]
#             print 'Species no improv: %s' % [s.no_improvement_age for s in self.__species] # species no improvement age

#             #for s in self.__species:
#             #    print s

#         # -------------------------- Producing new offspring -------------------------- #
#         new_population = [] # next generation's population

#         # Spawning new population
#         for s in self.__species:
#             new_population.extend(s.reproduce())

#         # ----------------------------#
#         # Controls under or overflow  #
#         # ----------------------------#
#         fill = (self.__popsize) - len(new_population)
#         if fill < 0: # overflow
#             if report: print '   Removing %d excess individual(s) from the new population' %-fill
#             # TODO: This is dangerous! I can't remove a species' representant!
#             new_population = new_population[:fill] # Removing the last added members

#         if fill > 0: # underflow
#             if report: print '   Producing %d more individual(s) to fill up the new population' %fill

#             # TODO:
#             # what about producing new individuals instead of reproducing?
#             # increasing diversity from time to time might help
#             while fill > 0:
#                 # Selects a random chromosome from population
#                 parent1 = random.choice(self.__population)
#                 # Search for a mate within the same species
#                 found = False
#                 for c in self:
#                     # what if c is parent1 itself?
#                     if c.species_id == parent1.species_id:
#                         child = parent1.crossover(c)
#                         new_population.append(child.mutate())
#                         found = True
#                         break
#                 if not found:
#                     # If no mate was found, just mutate it
#                     new_population.append(parent1.mutate())
#                 #new_population.append(chromosome.FFChromosome.create_fully_connected())
#                 fill -= 1

#         assert self.__popsize == len(new_population), 'Different population sizes!'
#         # Updates current population
#         self.__population = new_population[:]

#         if checkpoint_interval is not None and time.time() > t0 + 60*checkpoint_interval:
#             self.__create_checkpoint(report)
#             t0 = time.time() # updates the counter
#         elif checkpoint_generation is not None and self.__generation % checkpoint_generation == 0:
#             self.__create_checkpoint(report)
          end
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

#     def __resume_checkpoint(self, checkpoint):
#         """ Resumes the simulation from a previous saved point. """
#         try:
#             #file = open(checkpoint)
#             file = gzip.open(checkpoint)
#         except IOError:
#             raise
#         print 'Resuming from a previous point: %s' %checkpoint
#         # when unpickling __init__ is not called again
#         previous_pop = pickle.load(file)
#         self.__dict__ = previous_pop.__dict__

#         print 'Loading random state'
#         rstate = pickle.load(file)
#         random.setstate(rstate)
#         #random.jumpahead(1)
#         file.close()

#     def __create_checkpoint(self, report):
#         """ Saves the current simulation state. """
#         #from time import strftime
#         # get current time
#         #date = strftime("%Y_%m_%d_%Hh%Mm%Ss")
#         if report:
#             print 'Creating checkpoint file at generation: %d' %self.__generation

#         # dumps 'self'
#         #file = open('checkpoint_'+str(self.__generation), 'w')
#         file = gzip.open('checkpoint_'+str(self.__generation), 'w', compresslevel = 5)
#         # dumps the population
#         pickle.dump(self, file, protocol=2)
#         # dumps the current random state
#         pickle.dump(random.getstate(), file, protocol=2)
#         file.close()

