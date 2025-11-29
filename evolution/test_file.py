import genetic
import population

popul = [population.generate_random_individual(5, 2, 1) for i in range(10)]

species = genetic.create_species(popul, (1.2, 1.4, 0.6), 0.5)
print([[] for s in species])
print(len(species))