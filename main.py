import matplotlib.pyplot as plt
import networkx as nx
from beehive import Beehive

# Exemple de données (à remplacer par ton vrai champ de fleurs)
flowers = [(100,200), (150,800), (400,600), (700,300), (850,900)]
hive_position = (500,500)

def plot_best_path(best_bee, hive_position):
    x = [hive_position[0]] + [f[0] for f in best_bee.path] + [hive_position[0]]
    y = [hive_position[1]] + [f[1] for f in best_bee.path] + [hive_position[1]]
    plt.plot(x, y, marker="o")
    plt.title("Trajet de la meilleure abeille")
    plt.show()

def plot_history(beehive):
    plt.plot(beehive.history)
    plt.xlabel("Générations")
    plt.ylabel("Temps moyen")
    plt.title("Évolution du temps moyen par génération")
    plt.show()

def plot_genealogy(beehive, best_bee):
    G = nx.DiGraph()
    stack = [best_bee.id]
    while stack:
        bee_id = stack.pop()
        parents, gen = beehive.genealogy.get(bee_id, (None, None))
        if parents:
            for p in parents:
                if p is not None:
                    G.add_edge(p, bee_id)
                    stack.append(p)
    nx.draw(G, with_labels=True, node_size=500, font_size=8)
    plt.title("Arbre généalogique de la meilleure abeille")
    plt.show()

def compare_parameters(flowers, hive_position):
    configs = [
        {"mutation_rate": 0.01, "label": "mutation=0.01"},
        {"mutation_rate": 0.05, "label": "mutation=0.05"},
        {"mutation_rate": 0.1, "label": "mutation=0.1"},
    ]
    for cfg in configs:
        hive = Beehive(flowers, hive_position, mutation_rate=cfg["mutation_rate"], generations=30)
        hive.evolve()
        plt.plot(hive.history, label=cfg["label"])
    plt.xlabel("Générations")
    plt.ylabel("Temps moyen")
    plt.title("Comparaison des paramétrages")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    hive = Beehive(flowers, hive_position, mutation_rate=0.05, generations=50)
    best_bee = hive.evolve()

    plot_best_path(best_bee, hive_position)
    plot_history(hive)
    plot_genealogy(hive, best_bee)
    compare_parameters(flowers, hive_position)
