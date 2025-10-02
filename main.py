import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from beehive_2 import Beehive

def load_flowers_from_csv(csv_path):
    """Charge les coordonnées des fleurs depuis un CSV."""
    df = pd.read_csv(csv_path)
    #récuperes les coordonnées x et y
    flowers = list(zip(df['x'], df['y']))
    return flowers

# Example flowers (replace with the real coordinates)
flowers = load_flowers_from_csv("C:/Users/ndiay/OneDrive/Desktop/miel_abeille/venv/Champ_fleurs.csv")
hive_position = (500,500)

def plot_best_path(best_bee, hive_position):
    x = [hive_position[0]] + [f[0] for f in best_bee.path] + [hive_position[0]]
    y = [hive_position[1]] + [f[1] for f in best_bee.path] + [hive_position[1]]
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.scatter([hive_position[0]], [hive_position[1]], s=100, label="Hive")
    plt.title("Trajet de la meilleure abeille")
    plt.legend()
    plt.show()

def plot_history(beehive):
    plt.figure()
    plt.plot(beehive.history, label="Average distance")
    plt.plot(beehive.best_history, label="Best distance")
    plt.xlabel("Générations")
    plt.ylabel("Distance")
    plt.title("Évolution du temps moyen / meilleur par génération")
    plt.legend()
    plt.show()

def plot_genealogy(beehive, best_bee, max_nodes=50):
    G = nx.DiGraph()
    edges = beehive.get_genealogy_subgraph(best_bee.id, max_nodes=max_nodes)
    if not edges:
        print("Aucune relation généalogique retrouvée (vérifier genealogy).")
        return
    for p, c in edges:
        G.add_edge(p, c)
    # ensure queen node is present
    if beehive.queen:
        G.add_node(beehive.queen.id)
        # connect queen to any node that has queen as parent in edges
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8,6))
    nx.draw(G, pos, with_labels=True, node_size=400, font_size=8, arrowsize=12)
    plt.title("Arbre généalogique (limit nodes = {})".format(max_nodes))
    plt.show()

def compare_parameters(flowers, hive_position):
    configs = [
        {"mutation_rate": 0.01, "label": "mutation=0.01"},
        {"mutation_rate": 0.05, "label": "mutation=0.05"},
        {"mutation_rate": 0.1, "label": "mutation=0.1"},
    ]
    plt.figure()
    for cfg in configs:
        hive = Beehive(flowers, hive_position, population_size=101,
                       mutation_rate=cfg["mutation_rate"], generations=30)
        hive.evolve()
        plt.plot(hive.history, label=cfg["label"])
    plt.xlabel("Générations")
    plt.ylabel("Temps moyen")
    plt.title("Comparaison des paramétrages (mutation rate)")
    plt.legend()
    plt.show()

def compare_fitness_metrics(flowers, hive_position):
    modes = [
        {"fitness_mode":"inverse", "label":"inverse"},
        {"fitness_mode":"normalized", "label":"normalized"},
        {"fitness_mode":"exp", "label":"exp alpha=0.002", "alpha":0.002},
        {"fitness_mode":"ratio", "label":"ratio"},
    ]
    plt.figure()
    for m in modes:
        hive = Beehive(flowers, hive_position, population_size=101,
                       generations=40, fitness_mode=m.get("fitness_mode","inverse"),
                       alpha=m.get("alpha",0.01))
        hive.evolve()
        plt.plot(hive.best_history, label=m["label"])
    plt.xlabel("Générations")
    plt.ylabel("Best Distance")
    plt.title("Comparaison des métriques de fitness (best distance par génération)")
    plt.legend()
    plt.show()

def compare_reproduction_methods(flowers, hive_position):
    methods = [
        {"selection_method":"roulette", "label":"roulette"},
        {"selection_method":"tournament", "label":"tournament k=3", "tournament_size":3},
        {"selection_method":"rank", "label":"rank"},
    ]
    plt.figure()
    for m in methods:
        hive = Beehive(flowers, hive_position, population_size=101,
                       generations=40, selection_method=m.get("selection_method","roulette"),
                       tournament_size=m.get("tournament_size",3))
        hive.evolve()
        plt.plot(hive.best_history, label=m["label"])
    plt.xlabel("Générations")
    plt.ylabel("Best Distance")
    plt.title("Comparaison des méthodes de reproduction (best distance par génération)")
    plt.legend()
    plt.show()

def compare_crossover_methods(flowers, hive_position):
    methods = [
        {"crossover_method":"ox", "label":"OX (order)"},
        {"crossover_method":"pmx", "label":"PMX (partially mapped)"},
        {"crossover_method":"cx", "label":"CX (cycle)"},
    ]
    plt.figure()
    for m in methods:
        hive = Beehive(flowers, hive_position, population_size=101,
                       generations=30, crossover_method=m.get("crossover_method","ox"),
                       mutation_rate=0.05)
        hive.evolve()
        plt.plot(hive.best_history, label=m["label"])
    plt.xlabel("Generations")
    plt.ylabel("Best Distance")
    plt.title("Comparison of crossover methods")
    plt.legend()
    plt.show()
if __name__ == "__main__":
    hive = Beehive(flowers, hive_position, population_size=101, mutation_rate=0.05, generations=50)
    best_bee = hive.evolve()

    plot_best_path(best_bee, hive_position)
    plot_history(hive)
    plot_genealogy(hive, best_bee, max_nodes=30)
    compare_parameters(flowers, hive_position)
    compare_fitness_metrics(flowers, hive_position)
    compare_reproduction_methods(flowers, hive_position)
    compare_crossover_methods(flowers, hive_position)
