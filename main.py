import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from beehive2 import Beehive

def load_flowers_from_csv(csv_path):
    """Charge les coordonnées des fleurs depuis un CSV."""
    df = pd.read_csv(csv_path)
    #récuperes les coordonnées x et y
    flowers = list(zip(df['x'], df['y']))
    return flowers

# 
flowers = load_flowers_from_csv("C:/Users/ndiay/OneDrive/Desktop/miel_abeille/venv/Champ_fleurs.csv")
hive_position = (500,500)

import math

def plot_best_path(best_bee, hive_position):
    # --- Calcul des distances ruche → fleurs ---
    flower_distances = {
        flower: math.dist(hive_position, flower) for flower in best_bee.path
    }
    # Attribution d’un numéro selon la distance croissante
    sorted_flowers = sorted(flower_distances.items(), key=lambda x: x[1])
    flower_ids = {flower: i+1 for i, (flower, _) in enumerate(sorted_flowers)}

    # --- Coordonnées du trajet ---
    x = [hive_position[0]] + [f[0] for f in best_bee.path] + [hive_position[0]]
    y = [hive_position[1]] + [f[1] for f in best_bee.path] + [hive_position[1]]

    plt.figure(figsize=(10, 7))

    # Tracé du chemin avec flèches
    for i in range(len(x) - 1):
        plt.annotate("",
                     xy=(x[i+1], y[i+1]),
                     xytext=(x[i], y[i]),
                     arrowprops=dict(arrowstyle="->", color="blue", lw=1.5))

    # Points du chemin
    plt.plot(x, y, "o-", color="blue", alpha=0.6)

    # Ruche en jaune
    plt.scatter([hive_position[0]], [hive_position[1]],
                s=250, c="gold", edgecolors="black", label="Hive", zorder=3)

    # Numérotation des fleurs (selon distance à la ruche)
    for fx, fy in best_bee.path:
        fid = flower_ids[(fx, fy)]
        plt.text(fx, fy, str(fid), fontsize=9, ha="center", va="center",
                 bbox=dict(facecolor="white", edgecolor="black", boxstyle="circle,pad=0.3"))

    # Construire la liste du chemin (seulement les numéros)
    parcours = [str(flower_ids[(fx, fy)]) for fx, fy in best_bee.path]
    parcours_str = " → ".join(parcours)

    # Afficher la liste dans le graphe
    plt.gcf().text(0.1, 0.02, "Parcours: " + parcours_str,
                   fontsize=8, ha="left", va="bottom", wrap=True)

    plt.title("Trajet de la meilleure abeille (numérotation par distance à la ruche)")
    plt.legend()
    plt.axis("equal")
    plt.show()


def visualize_genealogy(beehive, figsize=(12, 8), save_as=None):
    """
    Affiche la généalogie des abeilles à partir de beehive.genealogy.
    """
    G = nx.DiGraph()
    
    for child_id, (parents, gen) in beehive.genealogy.items():
        G.add_node(child_id, generation=gen)
        if parents:
            for p in parents:
                if p is not None:
                    G.add_edge(p, child_id)
        elif child_id != "Queen":
            G.add_edge("Queen", child_id)
    
    pos = nx.multipartite_layout(G, subset_key="generation")
    
    plt.figure(figsize=figsize)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=600,
        node_color="skyblue",
        font_size=8,
        font_weight="bold",
        edge_color="gray",
        arrows=True,
        alpha=0.9,
    )
    
    plt.title(" Généalogie de la colonie", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_as:
        plt.savefig(save_as, dpi=300)
        print(f" Généalogie sauvegardée dans {save_as}")
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

def compare_parameters(flowers, hive_position):
    configs = [
        {"mutation_rate": 0.01, "label": "mutation=0.01"},
        {"mutation_rate": 0.05, "label": "mutation=0.05"},
        {"mutation_rate": 0.1, "label": "mutation=0.1"},
    ]
    plt.figure()
    for cfg in configs:
        beehive = Beehive(flowers, hive_position, population_size=101,
                       mutation_rate=cfg["mutation_rate"], generations=30)
        beehive.evolve()
        plt.plot(beehive.history, label=cfg["label"])
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
        beehive = Beehive(flowers, hive_position, population_size=101,
                       generations=30, crossover_method=m.get("crossover_method","ox"),
                       mutation_rate=0.05)
        beehive.evolve()
        plt.plot(beehive.best_history, label=m["label"])
    plt.xlabel("Generations")
    plt.ylabel("Best Distance")
    plt.title("Comparison of crossover methods")
    plt.legend()
    plt.show()
if __name__ == "__main__":
    beehive = Beehive(flowers, hive_position, population_size=101, mutation_rate=0.05, generations=50)
    best_bee = beehive.evolve()

    plot_best_path(best_bee, hive_position)
    plot_history(beehive)
    #visualize_genealogy(beehive, figsize=(12, 8), save_as="genealogy.png")
   
    compare_parameters(flowers, hive_position)
    compare_fitness_metrics(flowers, hive_position)
    compare_reproduction_methods(flowers, hive_position)
    compare_crossover_methods(flowers, hive_position)
