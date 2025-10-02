import random
import math

class Bee:
    _id_counter = 0

    def __init__(self, path=None, parents=None, is_queen=False, id=None):
        if id is None:
            self.id = Bee._id_counter
            Bee._id_counter += 1
        else:
            self.id = id
        self.path = path[:] if path is not None else []
        self.parents = parents
        self.fitness = None
        self.distance = None
        self.is_queen = is_queen

    def evaluate(self, hive_position, fitness_mode="inverse", alpha=0.01, reference=None, max_distance=None):
        if self.is_queen:
            self.distance = 0
            self.fitness = 0
            return self.fitness

        dist = 0
        current = hive_position
        for flower in self.path:
            dist += math.dist(current, flower)
            current = flower
        dist += math.dist(current, hive_position)
        self.distance = dist

        if fitness_mode == "inverse":
            self.fitness = 1.0 / (dist + 1e-9)
        elif fitness_mode == "normalized":
            if max_distance is None or max_distance == 0:
                self.fitness = 0.0
            else:
                self.fitness = max(0.0, (max_distance - dist) / max_distance)
        elif fitness_mode == "exp":
            self.fitness = math.exp(-alpha * dist)
        elif fitness_mode == "ratio":
            if reference is None or reference == 0:
                self.fitness = 0.0
            else:
                self.fitness = reference / dist
        else:
            self.fitness = 1.0 / (dist + 1e-9)
        return self.fitness


class Beehive:
    def __init__(self, flowers, hive_position=(500, 500),
                 population_size=101, mutation_rate=0.05, generations=500,
                 fitness_mode="inverse", selection_method="roulette",
                 tournament_size=3, elitism_rate=0.1, alpha=0.01,
                 crossover_method="ox"):
        self.flowers = flowers
        self.hive_position = hive_position
        self.population_size = population_size
        self.num_workers = max(1, population_size - 1)
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.fitness_mode = fitness_mode
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.elitism_rate = elitism_rate
        self.alpha = alpha
        self.crossover_method = crossover_method

        self.population = []
        self.queen = None
        self.genealogy = {}
        self.history = []
        self.best_history = []

    def init_population(self, seed_paths=None):
        self.queen = Bee(path=[], is_queen=True, id="Queen")
        self.genealogy[self.queen.id] = (None, 0)

        self.population = []
        for i in range(self.num_workers):
            if seed_paths and i < len(seed_paths):
                path = seed_paths[i][:]
            else:
                path = self.flowers[:]
                random.shuffle(path)
            bee = Bee(path)
            self.population.append(bee)
            self.genealogy[bee.id] = (bee.parents, 0)

    def evaluate_population(self):
        for bee in self.population:
            bee.distance = None
            bee.fitness = None

        distances = []
        for bee in self.population:
            dist = 0
            current = self.hive_position
            for flower in bee.path:
                dist += math.dist(current, flower)
                current = flower
            dist += math.dist(current, self.hive_position)
            bee.distance = dist
            distances.append(dist)

        max_distance = max(distances) if distances else 0
        best_distance = min(distances) if distances else 0

        for bee in self.population:
            bee.evaluate(self.hive_position, fitness_mode=self.fitness_mode,
                         alpha=self.alpha, reference=best_distance, max_distance=max_distance)

        avg_time = sum(b.distance for b in self.population) / len(self.population)
        best_time = min(b.distance for b in self.population)
        self.history.append(avg_time)
        self.best_history.append(best_time)
        return avg_time, best_time

    def select_one(self):
        if self.selection_method == "roulette":
            total_fitness = sum(bee.fitness for bee in self.population)
            if total_fitness == 0:
                return random.choice(self.population)
            pick = random.uniform(0, total_fitness)
            current = 0
            for bee in self.population:
                current += bee.fitness
                if current >= pick:
                    return bee
            return self.population[-1]
        elif self.selection_method == "tournament":
            contestants = random.sample(self.population, min(self.tournament_size, len(self.population)))
            return max(contestants, key=lambda b: b.fitness)
        elif self.selection_method == "rank":
            ranked = sorted(self.population, key=lambda b: b.fitness)
            n = len(ranked)
            weights = [i+1 for i in range(n)]
            total = sum(weights)
            pick = random.uniform(0, total)
            cur = 0
            for w, bee in zip(weights, ranked):
                cur += w
                if cur >= pick:
                    return bee
            return ranked[-1]
        else:
            return random.choice(self.population)

    def select_parents(self):
        p1 = self.select_one()
        p2 = self.select_one()
        return p1, p2

    def crossover(self, parent1, parent2):
        method = (self.crossover_method or "ox").lower()
        if method == "ox":
            return self._ox(parent1, parent2)
        elif method == "pmx":
            return self._pmx(parent1, parent2)
        elif method == "cx":
            return self._cx(parent1, parent2)
        else:
            return self._ox(parent1, parent2)

    def _ox(self, parent1, parent2):
        size = len(parent1.path)
        if size <= 1:
            return Bee(parent1.path[:], parents=(parent1.id, parent2.id))
        start, end = sorted(random.sample(range(size), 2))
        child_path = [None] * size
        child_path[start:end] = parent1.path[start:end]
        p2_genes = [g for g in parent2.path if g not in child_path]
        idx = 0
        for i in range(size):
            if child_path[i] is None:
                child_path[i] = p2_genes[idx]
                idx += 1
        return Bee(child_path, parents=(parent1.id, parent2.id))

    def _pmx(self, parent1, parent2):
        size = len(parent1.path)
        if size <= 1:
            return Bee(parent1.path[:], parents=(parent1.id, parent2.id))
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        # copy segment from parent1
        for i in range(start, end):
            child[i] = parent1.path[i]
        # map and fill
        for i in range(start, end):
            gene = parent2.path[i]
            if gene not in child:
                pos = i
                val = gene
                while True:
                    val_p1 = parent1.path[pos]
                    pos = parent2.path.index(val_p1)
                    if child[pos] is None:
                        child[pos] = gene
                        break
        # fill remaining spots with parent2 genes
        for i in range(size):
            if child[i] is None:
                child[i] = parent2.path[i]
        return Bee(child, parents=(parent1.id, parent2.id))

    def _cx(self, parent1, parent2):
        size = len(parent1.path)
        child = [None] * size
        cycles = 0
        while None in child:
            # find first index not assigned
            start_idx = next(i for i, v in enumerate(child) if v is None)
            idx = start_idx
            cycle_indices = []
            while True:
                cycle_indices.append(idx)
                val = parent2.path[idx]
                idx = parent1.path.index(val)
                if idx == start_idx:
                    break
            # copy depending on cycle parity
            if cycles % 2 == 0:
                for k in cycle_indices:
                    child[k] = parent1.path[k]
            else:
                for k in cycle_indices:
                    child[k] = parent2.path[k]
            cycles += 1
        return Bee(child, parents=(parent1.id, parent2.id))

    def mutate(self, bee):
        if random.random() < self.mutation_rate and len(bee.path) >= 2:
            i, j = random.sample(range(len(bee.path)), 2)
            bee.path[i], bee.path[j] = bee.path[j], bee.path[i]

    def evolve(self):
        self.history = []
        self.best_history = []
        self.init_population()
        self.evaluate_population()

        for gen in range(1, self.generations + 1):
            self.population.sort(key=lambda b: b.fitness, reverse=True)
            elitism_count = int(self.elitism_rate * len(self.population))
            elites = self.population[:elitism_count]
            survivors = self.population[: max(elitism_count, len(self.population) // 2)]
            children = []
            while len(survivors) + len(children) < len(self.population):
                p1, p2 = self.select_parents()
                child = self.crossover(p1, p2)
                self.mutate(child)
                children.append(child)
                self.genealogy[child.id] = (child.parents, gen)
            self.population = elites + survivors + children
            self.population = self.population[:len(survivors) + len(children) + elitism_count]
            self.evaluate_population()

        best_bee = max(self.population, key=lambda b: b.fitness)
        return best_bee

    def get_genealogy_subgraph(self, best_bee_id, max_nodes=50):
        edges = []
        visited = set()
        stack = [best_bee_id]
        while stack and len(visited) < max_nodes:
            bee_id = stack.pop()
            if bee_id in visited:
                continue
            visited.add(bee_id)
            parents_gen = self.genealogy.get(bee_id)
            if parents_gen:
                parents, gen = parents_gen
                if parents:
                    for p in parents:
                        if p is not None:
                            edges.append((p, bee_id))
                            stack.append(p)
                else:
                    if self.queen and bee_id != self.queen.id:
                        p0 = self.genealogy.get(bee_id, (None, None))[1]
                        if p0 == 0:
                            edges.append((self.queen.id, bee_id))
                            stack.append(self.queen.id)
        return edges
