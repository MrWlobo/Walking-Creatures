import genetic
import population
from visualization import Visualization
import matplotlib.pyplot as plt

popul = [population.generate_random_individual(5, 2, 1) for i in range(10)]

species = genetic.create_species(popul, (1.2, 1.4, 0.6), 0.5)

vis = Visualization("visualizations")

fig, ax = plt.subplots(1, 2)
vis.visualize_network(popul[1], ax[0], "test", False, True)