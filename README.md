# Genetic Algorithm (GA) Implementation 🧬⚙️

Evolutionary computation provides robust solutions for complex search and optimization problems where traditional gradient-based methods fail. This repository contains a foundational, from-scratch implementation of a classic **Genetic Algorithm (GA)**.

Built with clean architecture in mind, the project decouples the core evolutionary logic from the experimental execution, making the algorithm highly modular and easy to integrate into larger Machine Learning or optimization pipelines.

## 🎯 Project Overview
The core mechanics of the algorithm (Population Initialization, Fitness Evaluation, Selection, Crossover, and Mutation) are encapsulated within a standalone Python script (`genetic_algorithm.py`). 

The `ag_experimentacion.ipynb` notebook serves as the testing ground, where the algorithm is instantiated, hyperparameter tuning is conducted (e.g., mutation rates, population sizing), and the evolutionary progress is tracked across generations.

## 🚀 Key Features
* **Built From Scratch:** No black-box libraries used for the evolutionary process. The core operators are implemented in pure Python to demonstrate a deep understanding of bio-inspired heuristics.
* **Modular Operators:** Easily swappable functions for different types of crossover (e.g., single-point, uniform) and mutation strategies.
* **Decoupled Experimentation:** Clean separation of concerns between algorithm definition (`.py`) and execution/visualization (`.ipynb`).
* **Convergence Tracking:** Capable of tracking the best, worst, and average fitness scores per generation to visualize the algorithm's convergence.

## 🛠️ Tech Stack
* **Language:** Python
* **Computation:** `numpy`
* **Visualization:** `matplotlib`


## ⚙️ How to Run
1. Clone this repository.
2. Ensure you have the required dependencies (`numpy`, `matplotlib`).
3. Review the evolutionary logic in `genetic_algorithm.py`.
