import random
import math

class Bee:
    _id_counter = 0

    def __init__(self, path, parents=None):
        self.id = Bee._id_counter
        Bee._id_counter += 1
        self.path = path[:]  # liste de positions (x,y)
        self.parents = parents  # tuple (id_parent1, id_parent2)
        self.fitness = None
        self.distance = None

    def evaluate(self, hive_position):
        """Calcule la distance totale du parcours (depuis/vers la ruche)."""
        dist = 0
        current = hive_position
        for flower in self.path:
            dist += math.dist(current, flower)
            current = flower
        dist += math.dist(current, hive_position)  # retour à la ruche
        self.distance = dist
        self.fitness = 1 / dist
        return self.fitness


class Beehive:
    def __init__(self, flowers, hive_position=(500, 500),
                 population_size=100, mutation_rate=0.05, generations=500):
        self.flowers = flowers
        self.hive_position = hive_position
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.population = []
        self.genealogy = {}  # id -> (parents, generation)
        self.history = []    # temps moyen par génération

    def init_population(self):
        """Crée la population initiale (permutations aléatoires des fleurs)."""
        self.population = []
        for _ in range(self.population_size):
            path = self.flowers[:]
            random.shuffle(path)
            bee = Bee(path)
            bee.evaluate(self.hive_position)
            self.population.append(bee)
            self.genealogy[bee.id] = (bee.parents, 0)

    def select_parents(self):
        """Sélectionne 2 abeilles par roulette wheel (selon fitness)."""
        total_fitness = sum(bee.fitness for bee in self.population)
        pick = random.uniform(0, total_fitness)
        current = 0
        for bee in self.population:
            current += bee.fitness
            if current >= pick:
                parent1 = bee
                break

        pick = random.uniform(0, total_fitness)
        current = 0
        for bee in self.population:
            current += bee.fitness
            if current >= pick:
                parent2 = bee
                break

        return parent1, parent2

    def crossover(self, parent1, parent2):
        """Croisement en utilisant un ordre partiel (OX)."""
        start, end = sorted(random.sample(range(len(parent1.path)), 2))
        child_path = [None] * len(parent1.path)
        child_path[start:end] = parent1.path[start:end]

        p2_genes = [g for g in parent2.path if g not in child_path]
        idx = 0
        for i in range(len(child_path)):
            if child_path[i] is None:
                child_path[i] = p2_genes[idx]
                idx += 1

        return Bee(child_path, parents=(parent1.id, parent2.id))

    def mutate(self, bee):
        """Mutation par échange aléatoire de deux positions."""
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(bee.path)), 2)
            bee.path[i], bee.path[j] = bee.path[j], bee.path[i]

    def evolve(self):
        """Fait évoluer la population pendant N générations."""
        self.init_population()

        for gen in range(1, self.generations + 1):
            # Évaluation
            for bee in self.population:
                bee.evaluate(self.hive_position)

            avg_time = sum(b.distance for b in self.population) / len(self.population)
            self.history.append(avg_time)

            # Sélection des meilleurs
            self.population.sort(key=lambda b: b.fitness, reverse=True)
            survivors = self.population[: self.population_size // 2]

            # Nouvelle génération
            children = []
            while len(survivors) + len(children) < self.population_size:
                p1, p2 = self.select_parents()
                child = self.crossover(p1, p2)
                self.mutate(child)
                child.evaluate(self.hive_position)
                self.genealogy[child.id] = (child.parents, gen)
                children.append(child)

            self.population = survivors + children

        # Retourne la meilleure abeille
        best_bee = max(self.population, key=lambda b: b.fitness)
        return best_bee
