import random
import math
import networkx as nx
from graphviz import Digraph


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
        """Calcule la distance du parcours et la fitness selon le mode choisi."""
        if self.is_queen:
            self.distance = 0
            self.fitness = 0
            return self.fitness

        # Calcul unique de la distance totale
        dist = 0
        current = hive_position
        for flower in self.path:
            dist += math.dist(current, flower)
            current = flower
        dist += math.dist(current, hive_position)
        self.distance = dist

        # Calcul de la fitness
        if fitness_mode == "inverse":
            self.fitness = 1.0 / (dist + 1e-9)
        elif fitness_mode == "normalized":
            if max_distance:
                self.fitness = max(0.0, (max_distance - dist) / max_distance)
            else:
                self.fitness = 0.0
        elif fitness_mode == "exp":
            self.fitness = math.exp(-alpha * dist)
        elif fitness_mode == "ratio":
            if reference:
                self.fitness = reference / dist
            else:
                self.fitness = 0.0
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
        """Crée la reine et les abeilles initiales."""
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
        """Évalue chaque abeille de la population."""
        distances = []
        for bee in self.population:
            bee.evaluate(self.hive_position, fitness_mode=self.fitness_mode,
                         alpha=self.alpha)
            distances.append(bee.distance)

        max_distance = max(distances)
        best_distance = min(distances)

        # recalcul fitness avec les références globales
        for bee in self.population:
            bee.evaluate(self.hive_position, fitness_mode=self.fitness_mode,
                         alpha=self.alpha, reference=best_distance, max_distance=max_distance)

        avg_time = sum(b.distance for b in self.population) / len(self.population)
        best_time = min(b.distance for b in self.population)
        self.history.append(avg_time)
        self.best_history.append(best_time)
        return avg_time, best_time

    def select_one(self):
        """Sélectionne une abeille selon la méthode définie."""
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
        elif self.selection_method == "tournament":
            contestants = random.sample(self.population, min(self.tournament_size, len(self.population)))
            return max(contestants, key=lambda b: b.fitness)
        elif self.selection_method == "rank":
            ranked = sorted(self.population, key=lambda b: b.fitness)
            n = len(ranked)
            weights = [i + 1 for i in range(n)]
            total = sum(weights)
            pick = random.uniform(0, total)
            cur = 0
            for w, bee in zip(weights, ranked):
                cur += w
                if cur >= pick:
                    return bee
        return random.choice(self.population)

    def select_parents(self):
        """Sélectionne deux parents distincts."""
        p1 = self.select_one()
        p2 = self.select_one()
        attempts = 0
        while p2.id == p1.id and attempts < 10:
            p2 = self.select_one()
            attempts += 1
        return p1, p2

    # --- Les méthodes de crossover
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
        """Exécute l'évolution complète et retourne la meilleure abeille."""
        self.history.clear()
        self.best_history.clear()
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
            self.population = self.population[:len(self.population)]
            self.evaluate_population()

        best_bee = max(self.population, key=lambda b: b.fitness)
        return best_bee

    def export_genealogy_graph(self, filename="genealogy_graph.gexf"):
        """Exporte la généalogie au format NetworkX (GEXF)."""
        G = nx.DiGraph()
        for child_id, (parents, gen) in self.genealogy.items():
            G.add_node(child_id, generation=gen)
            if parents:
                for p in parents:
                    if p is not None:
                        G.add_edge(p, child_id)
            elif child_id != "Queen":
                G.add_edge("Queen", child_id)
        nx.write_gexf(G, filename)
        print(f"✅ Généalogie exportée vers {filename}")
        return G

    def visualize_evolution_flow(self, filename="evolution_flow"):
        """Crée un diagramme de flux de l'évolution (Graphviz)."""
        dot = Digraph(comment="Beehive Evolution Process", format="png")
        dot.attr(rankdir="LR", size="8,5")

        dot.node("A", "Initialisation de la ruche", shape="box")
        dot.node("B", "Évaluation de la population", shape="box")
        dot.node("C", "Sélection des parents", shape="box")
        dot.node("D", "Crossover (OX / PMX / CX)", shape="box")
        dot.node("E", "Mutation", shape="box")
        dot.node("F", "Nouvelle génération", shape="box")
        dot.node("G", "Évaluation + Statistiques", shape="box")
        dot.node("H", "Meilleure abeille trouvée", shape="ellipse", color="green")

        dot.edges(["AB", "BC", "CD", "DE", "EF", "FG"])
        dot.edge("G", "C", label="Répéter jusqu’à max générations")
        dot.edge("G", "H", label="Fin")

        dot.render(filename, cleanup=True)
        print(f" Diagramme de flux enregistré sous {filename}.png")
