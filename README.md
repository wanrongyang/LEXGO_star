# Python Implementation for LEXGO* 
This is the python implementation based on the paper: Multiobjective shortest path problems with lexicographic goal-based preferences. And this repository contains the Python implementation of the algorithm presented in the paper "[Multiobjective shortest path problems with lexicographic goal-based preferences](https://www.sciencedirect.com/science/article/pii/S0377221714004159?via%3Dihub)" by Francisco Javier Pulido, Lawrence Mandow, and José Luis Pérez de la Cruz. This implementation aims to reproduce the results and facilitate further research and applications.

## 📚 Paper Details

- **Title:** Multiobjective Shortest Path Problems with Lexicographic Goal-Based Preferences  
- **Authors:** Francisco Javier Pulido, Lawrence Mandow, José Luis Pérez de la Cruz  
- **Journal:** European Journal of Operational Research  
- **Volume:** 239  
- **Pages:** 89–101  
- **Published Year:** 2014  
- **DOI:** [10.1016/j.ejor.2014.05.008](http://dx.doi.org/10.1016/j.ejor.2014.05.008)

## 🧪 Before you start
During the implementation, it seems that there is a small mistake in the original equation 5:

$$
\vec{y} \prec_G \vec{y}' \iff \vec{d}(\vec{y}) \prec_L \vec{d}(\vec{y}') \lor (\vec{d}(\vec{y}) = \vec{d}(\vec{y}') \land \vec{y} \prec \vec{y}')
$$

However, the following equation seems to be right and the python code also follows the following equation and then get the same results based on the proposed example. I am not sure about this, maybe you can check it while you use this.

$$
\vec{y} \prec_G \vec{y}' \iff \vec{d}(\vec{y}) \prec_L \vec{d}(\vec{y}') \lor (\vec{d}(\vec{y}) = \vec{d}(\vec{y}') \land \vec{y} \prec_L \vec{y}')
$$

## 📝 Introduction

Multiobjective shortest path problems are significantly more challenging than single-objective ones due to the exponential growth in the number of label expansions with solution depth. This repository provides a Python implementation of a new exact label-setting algorithm that returns the subset of Pareto optimal paths satisfying a set of lexicographic goals or minimizing deviation from these goals.

## 🚀 Features

- **NetworkX Integration:** Utilizes the powerful NetworkX library for graph operations and manipulation.
- **Multiobjective Optimization:** Handles multiple objectives efficiently using lexicographic goal preferences.
- **Customizable Goals:** Allows users to define their own goals and preferences easily.
- **Performance Optimization:** Implements advanced pruning and filtering techniques to enhance performance.
- **Easy-to-Use Class:** Provides a straightforward interface for defining graphs, goals, and running the algorithm.
- **Extensive Documentation:** Includes detailed documentation and examples for quick setup and usage.
- **Get Shortest Distances:** By running the code, you can easily get the shortest distances in multiobjective directed graph.

## 🛠️ Installation

You can just simplely donwload this code and run it by following the usage instruction, and before that, you should install essential libraries.

```bash
pip install -r requirements.txt
```
## 🎉 Usage

Running the example in the paper to start the algorithm

```bash
def main():

    # define a directed graph
    graph = nx.DiGraph()

    # Given nodes and edges (example in the original paper)
    edges = [
             ('s', 'n1', {'weight': (2, 2, 2)}),
             ('s', 'n3', {'weight': (7, 6, 2)}),
             ('s', 'n2', {'weight': (3, 3, 6)}),
             ('n1', 'n3', {'weight': (3, 3, 3)}),
             ('n1', 't', {'weight': (8, 6, 8)}),
             ('n3', 't', {'weight': (5, 4, 2)}),
             ('n2', 'n3', {'weight': (2, 2, 2)}),
             ('n2', 't', {'weight': (9, 5, 2)})
            ]

    graph.add_edges_from(edges)

    # Attribute number starts from 0
    # The string '0' in lgp represents the first priority level, and the string '1' represents the second priority.
    # Define the start node, goal node, and lexicographic goal preference (lgp)
    start = 's'
    goal = 't'
    lgp = {'0': [(0, 10, 0.5), (1, 10, 0.5)], '1': [(2, 10, 1)]}

    # Initialize the LEXGO* algorithm
    lexgo = LEXGO(graph=graph, start_node=start, destination=goal, lex_preference=lgp)

    # Run the LEXGO* algorithm

    lexgo.running()


print("Running LEXGO algorithm...")

if __name__ == '__main__':
    main()
```
## 👍 Acknowledgements
Thanks for the essential support from my supervisor [Dominik](https://scholar.google.com/citations?user=-HObEAYAAAAJ) and [Centre for Doctorial Training in Distributed Algorithm (DA CDT)](https://www.liverpool.ac.uk/distributed-algorithms-cdt/)

## 📬 Contact
For any questions or suggestions, feel free to open an issue or contact us directly at Wanrong.Yang@liverpool.ac.uk. If you have better idea on how to improve the code, or you may find some mistakes to the code, please do let me know!

