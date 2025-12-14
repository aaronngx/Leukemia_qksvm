#!/usr/bin/env python3
"""
ENSGA (Elitist Non-Dominated Sorting Genetic Algorithm) for QSVM Optimization

This module implements QSVM-NDSGOA: Enhanced QSVM with multi-objective optimization
using ENSGA to solve the QSVM dual problem and optimize kernel parameters.

Key Features:
1. Non-dominated sorting (Pareto ranking)
2. Crowding distance for diversity preservation
3. Genetic operators (crossover, mutation)
4. Multi-objective optimization (accuracy, margin, complexity)

Reference: Algorithm 1 - QSVM-NDSGOA Pipeline
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass, field
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Individual:
    """Represents a solution in the population."""
    genes: np.ndarray                    # Solution parameters (alpha values or kernel params)
    objectives: np.ndarray = None        # Objective function values
    rank: int = 0                        # Pareto rank (0 = best front)
    crowding_distance: float = 0.0       # Diversity measure
    
    def __post_init__(self):
        if self.objectives is None:
            self.objectives = np.array([])
    
    def dominates(self, other: 'Individual') -> bool:
        """Check if this individual dominates another (Pareto dominance)."""
        if self.objectives is None or other.objectives is None:
            return False
        
        # Dominates if: at least as good in all objectives AND strictly better in at least one
        at_least_as_good = np.all(self.objectives <= other.objectives)
        strictly_better = np.any(self.objectives < other.objectives)
        return at_least_as_good and strictly_better


@dataclass
class Population:
    """Collection of individuals."""
    individuals: List[Individual] = field(default_factory=list)
    
    def __len__(self):
        return len(self.individuals)
    
    def __getitem__(self, idx):
        return self.individuals[idx]
    
    def append(self, ind: Individual):
        self.individuals.append(ind)
    
    def extend(self, inds: List[Individual]):
        self.individuals.extend(inds)


# =============================================================================
# NON-DOMINATED SORTING
# =============================================================================

def non_dominated_sort(population: Population) -> List[List[int]]:
    """
    Perform non-dominated sorting on the population.
    
    Returns:
        List of fronts, where each front is a list of individual indices.
        Front 0 contains the Pareto-optimal solutions.
    """
    n = len(population)
    
    # domination_count[i] = number of individuals that dominate i
    domination_count = [0] * n
    
    # dominated_set[i] = set of individuals that i dominates
    dominated_set = [[] for _ in range(n)]
    
    fronts = [[]]  # Start with empty first front
    
    # Compare all pairs
    for i in range(n):
        for j in range(i + 1, n):
            if population[i].dominates(population[j]):
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif population[j].dominates(population[i]):
                dominated_set[j].append(i)
                domination_count[i] += 1
    
    # Find first front (non-dominated individuals)
    for i in range(n):
        if domination_count[i] == 0:
            population[i].rank = 0
            fronts[0].append(i)
    
    # Build subsequent fronts
    current_front = 0
    while fronts[current_front]:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    population[j].rank = current_front + 1
                    next_front.append(j)
        
        current_front += 1
        fronts.append(next_front)
    
    # Remove empty last front
    if not fronts[-1]:
        fronts.pop()
    
    return fronts


# =============================================================================
# CROWDING DISTANCE
# =============================================================================

def calculate_crowding_distance(population: Population, front: List[int]):
    """
    Calculate crowding distance for individuals in a front.
    
    Crowding distance measures the density of solutions surrounding a particular
    solution, used to maintain diversity in the population.
    """
    if len(front) <= 2:
        for i in front:
            population[i].crowding_distance = float('inf')
        return
    
    # Initialize distances
    for i in front:
        population[i].crowding_distance = 0.0
    
    n_objectives = len(population[front[0]].objectives)
    
    for m in range(n_objectives):
        # Sort front by objective m
        sorted_front = sorted(front, key=lambda i: population[i].objectives[m])
        
        # Boundary points get infinite distance
        population[sorted_front[0]].crowding_distance = float('inf')
        population[sorted_front[-1]].crowding_distance = float('inf')
        
        # Calculate range for normalization
        obj_min = population[sorted_front[0]].objectives[m]
        obj_max = population[sorted_front[-1]].objectives[m]
        obj_range = obj_max - obj_min
        
        if obj_range == 0:
            continue
        
        # Calculate crowding distance for middle points
        for i in range(1, len(sorted_front) - 1):
            prev_obj = population[sorted_front[i - 1]].objectives[m]
            next_obj = population[sorted_front[i + 1]].objectives[m]
            population[sorted_front[i]].crowding_distance += (next_obj - prev_obj) / obj_range


# =============================================================================
# GENETIC OPERATORS
# =============================================================================

def sbx_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    eta: float = 20.0,
    prob: float = 0.9
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulated Binary Crossover (SBX).
    
    Args:
        parent1, parent2: Parent gene vectors
        eta: Distribution index (larger = children closer to parents)
        prob: Crossover probability
    
    Returns:
        Two offspring gene vectors
    """
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    if np.random.random() > prob:
        return child1, child2
    
    for i in range(len(parent1)):
        if np.random.random() > 0.5:
            continue
        
        if abs(parent1[i] - parent2[i]) < 1e-14:
            continue
        
        y1 = min(parent1[i], parent2[i])
        y2 = max(parent1[i], parent2[i])
        
        rand = np.random.random()
        
        # Calculate beta
        beta = 1.0 + (2.0 * y1) / (y2 - y1 + 1e-14)
        alpha = 2.0 - beta ** (-(eta + 1.0))
        
        if rand <= 1.0 / alpha:
            betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
        
        c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
        c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
        
        child1[i] = c1
        child2[i] = c2
    
    return child1, child2


def polynomial_mutation(
    genes: np.ndarray,
    bounds: Tuple[float, float],
    eta: float = 20.0,
    prob: float = None
) -> np.ndarray:
    """
    Polynomial mutation.
    
    Args:
        genes: Gene vector to mutate
        bounds: (lower, upper) bounds for genes
        eta: Distribution index
        prob: Mutation probability per gene (default: 1/n_genes)
    
    Returns:
        Mutated gene vector
    """
    mutated = genes.copy()
    n = len(genes)
    
    if prob is None:
        prob = 1.0 / n
    
    lower, upper = bounds
    
    for i in range(n):
        if np.random.random() > prob:
            continue
        
        y = mutated[i]
        delta1 = (y - lower) / (upper - lower + 1e-14)
        delta2 = (upper - y) / (upper - lower + 1e-14)
        
        rand = np.random.random()
        mut_pow = 1.0 / (eta + 1.0)
        
        if rand < 0.5:
            xy = 1.0 - delta1
            val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
            deltaq = val ** mut_pow - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
            deltaq = 1.0 - val ** mut_pow
        
        mutated[i] = y + deltaq * (upper - lower)
        mutated[i] = np.clip(mutated[i], lower, upper)
    
    return mutated


def tournament_selection(population: Population, k: int = 2) -> Individual:
    """
    Binary tournament selection based on rank and crowding distance.
    """
    candidates = np.random.choice(len(population), size=k, replace=False)
    
    best = candidates[0]
    for idx in candidates[1:]:
        # Prefer lower rank (better Pareto front)
        if population[idx].rank < population[best].rank:
            best = idx
        # If same rank, prefer larger crowding distance (more diverse)
        elif population[idx].rank == population[best].rank:
            if population[idx].crowding_distance > population[best].crowding_distance:
                best = idx
    
    return population[best]


# =============================================================================
# QSVM OBJECTIVE FUNCTIONS
# =============================================================================

class QSVMObjectives:
    """
    Multi-objective functions for QSVM optimization.
    
    Objectives (minimization):
    1. Classification error (1 - accuracy)
    2. Model complexity (number of support vectors / n_samples)
    3. Inverse margin (1 / margin width)
    """
    
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        kernel_matrix: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        cv_folds: int = 3,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.kernel_matrix = kernel_matrix
        self.X_val = X_val
        self.y_val = y_val
        self.cv_folds = cv_folds
        self.n_samples = len(y_train)
    
    def evaluate(self, C: float, gamma: float = None) -> np.ndarray:
        """
        Evaluate QSVM with given parameters.
        
        Args:
            C: SVM regularization parameter
            gamma: Kernel parameter (optional, for RBF-like scaling)
        
        Returns:
            Array of objective values [error, complexity, inverse_margin]
        """
        # Scale kernel if gamma provided
        if gamma is not None:
            K = self.kernel_matrix * gamma
        else:
            K = self.kernel_matrix
        
        # Train SVM
        svm = SVC(kernel='precomputed', C=C, probability=False)
        
        try:
            svm.fit(K, self.y_train)
            
            # Objective 1: Classification error
            if self.X_val is not None and self.y_val is not None:
                # Use validation set
                K_val = K[:len(self.y_val), :]  # Adjust if separate kernel needed
                y_pred = svm.predict(K)
                error = 1.0 - accuracy_score(self.y_train, y_pred)
            else:
                # Use cross-validation
                scores = cross_val_score(svm, K, self.y_train, cv=self.cv_folds)
                error = 1.0 - np.mean(scores)
            
            # Objective 2: Model complexity (support vector ratio)
            n_support = len(svm.support_)
            complexity = n_support / self.n_samples
            
            # Objective 3: Inverse margin (approximation)
            # margin ∝ 1/||w|| ∝ 1/sqrt(sum(alpha_i * alpha_j * K_ij))
            dual_coef = np.abs(svm.dual_coef_[0])
            support_idx = svm.support_
            K_support = K[np.ix_(support_idx, support_idx)]
            w_norm_sq = np.dot(dual_coef, np.dot(K_support, dual_coef))
            inverse_margin = np.sqrt(w_norm_sq + 1e-8)
            
            return np.array([error, complexity, inverse_margin])
            
        except Exception as e:
            # Return worst objectives on failure
            return np.array([1.0, 1.0, 1e6])


# =============================================================================
# ENSGA OPTIMIZER
# =============================================================================

class ENSGA:
    """
    Elitist Non-Dominated Sorting Genetic Algorithm (ENSGA/NSGA-II).
    
    For QSVM optimization, optimizes:
    - C: SVM regularization parameter
    - gamma: Kernel scaling parameter
    
    Multi-objective: Minimize [error, complexity, inverse_margin]
    """
    
    def __init__(
        self,
        pop_size: int = 50,
        n_generations: int = 100,
        crossover_prob: float = 0.9,
        mutation_prob: float = None,
        crossover_eta: float = 20.0,
        mutation_eta: float = 20.0,
        bounds: Tuple[Tuple[float, float], ...] = ((0.01, 100.0), (0.001, 10.0)),
        verbose: bool = True,
    ):
        """
        Initialize ENSGA optimizer.
        
        Args:
            pop_size: Population size (should be even)
            n_generations: Maximum generations
            crossover_prob: Crossover probability
            mutation_prob: Mutation probability per gene
            crossover_eta: SBX distribution index
            mutation_eta: Polynomial mutation distribution index
            bounds: Parameter bounds [(C_min, C_max), (gamma_min, gamma_max)]
            verbose: Print progress
        """
        self.pop_size = pop_size if pop_size % 2 == 0 else pop_size + 1
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.crossover_eta = crossover_eta
        self.mutation_eta = mutation_eta
        self.bounds = bounds
        self.n_genes = len(bounds)
        self.verbose = verbose
        
        # Results
        self.pareto_front: List[Individual] = []
        self.history: List[Dict] = []
    
    def _initialize_population(self) -> Population:
        """Create random initial population."""
        pop = Population()
        
        for _ in range(self.pop_size):
            genes = np.array([
                np.random.uniform(self.bounds[i][0], self.bounds[i][1])
                for i in range(self.n_genes)
            ])
            pop.append(Individual(genes=genes))
        
        return pop
    
    def _evaluate_population(self, population: Population, objectives: QSVMObjectives):
        """Evaluate all individuals in population."""
        for ind in population.individuals:
            C = ind.genes[0]
            gamma = ind.genes[1] if self.n_genes > 1 else None
            ind.objectives = objectives.evaluate(C, gamma)
    
    def _create_offspring(self, population: Population) -> Population:
        """Create offspring population using genetic operators."""
        offspring = Population()
        
        while len(offspring) < self.pop_size:
            # Selection
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            
            # Crossover
            child1_genes, child2_genes = sbx_crossover(
                parent1.genes, parent2.genes,
                eta=self.crossover_eta,
                prob=self.crossover_prob
            )
            
            # Mutation
            gene_bounds = (
                min(b[0] for b in self.bounds),
                max(b[1] for b in self.bounds)
            )
            
            child1_genes = polynomial_mutation(
                child1_genes, gene_bounds,
                eta=self.mutation_eta,
                prob=self.mutation_prob
            )
            child2_genes = polynomial_mutation(
                child2_genes, gene_bounds,
                eta=self.mutation_eta,
                prob=self.mutation_prob
            )
            
            # Ensure bounds
            for i in range(self.n_genes):
                child1_genes[i] = np.clip(child1_genes[i], self.bounds[i][0], self.bounds[i][1])
                child2_genes[i] = np.clip(child2_genes[i], self.bounds[i][0], self.bounds[i][1])
            
            offspring.append(Individual(genes=child1_genes))
            offspring.append(Individual(genes=child2_genes))
        
        return offspring
    
    def _select_survivors(self, combined: Population) -> Population:
        """Select next generation using elitist non-dominated sorting."""
        # Non-dominated sorting
        fronts = non_dominated_sort(combined)
        
        # Calculate crowding distance for each front
        for front in fronts:
            calculate_crowding_distance(combined, front)
        
        # Select individuals for next generation
        new_pop = Population()
        front_idx = 0
        
        while len(new_pop) + len(fronts[front_idx]) <= self.pop_size:
            for i in fronts[front_idx]:
                new_pop.append(combined[i])
            front_idx += 1
            
            if front_idx >= len(fronts):
                break
        
        # Fill remaining slots from last front using crowding distance
        if len(new_pop) < self.pop_size and front_idx < len(fronts):
            last_front = fronts[front_idx]
            # Sort by crowding distance (descending)
            last_front_sorted = sorted(
                last_front,
                key=lambda i: combined[i].crowding_distance,
                reverse=True
            )
            
            remaining = self.pop_size - len(new_pop)
            for i in last_front_sorted[:remaining]:
                new_pop.append(combined[i])
        
        return new_pop
    
    def optimize(self, objectives: QSVMObjectives) -> Dict[str, Any]:
        """
        Run ENSGA optimization.
        
        Args:
            objectives: QSVMObjectives instance for evaluation
        
        Returns:
            Dict with pareto_front, best_solution, history
        """
        if self.verbose:
            print("="*60)
            print("ENSGA OPTIMIZATION FOR QSVM")
            print("="*60)
            print(f"Population: {self.pop_size}")
            print(f"Generations: {self.n_generations}")
            print(f"Parameters: {self.n_genes}")
            print(f"Bounds: {self.bounds}")
        
        # Initialize
        P_t = self._initialize_population()
        self._evaluate_population(P_t, objectives)
        
        # Main loop
        for gen in range(self.n_generations):
            # Create offspring Q_t
            Q_t = self._create_offspring(P_t)
            self._evaluate_population(Q_t, objectives)
            
            # Merge: R_t = P_t ∪ Q_t
            R_t = Population()
            R_t.extend(P_t.individuals)
            R_t.extend(Q_t.individuals)
            
            # Select next generation: P_{t+1}
            P_t = self._select_survivors(R_t)
            
            # Record history
            fronts = non_dominated_sort(P_t)
            pareto_front = [P_t[i] for i in fronts[0]]
            
            best_error = min(ind.objectives[0] for ind in pareto_front)
            avg_error = np.mean([ind.objectives[0] for ind in P_t.individuals])
            
            self.history.append({
                'generation': gen + 1,
                'best_error': best_error,
                'avg_error': avg_error,
                'pareto_size': len(pareto_front),
            })
            
            if self.verbose and (gen + 1) % 10 == 0:
                print(f"Gen {gen+1:3d}: Best Error={best_error:.4f}, "
                      f"Avg Error={avg_error:.4f}, Pareto Size={len(pareto_front)}")
        
        # Final pareto front
        fronts = non_dominated_sort(P_t)
        self.pareto_front = [P_t[i] for i in fronts[0]]
        
        # Find best solution (lowest error on pareto front)
        best_idx = np.argmin([ind.objectives[0] for ind in self.pareto_front])
        best_solution = self.pareto_front[best_idx]
        
        if self.verbose:
            print("="*60)
            print("OPTIMIZATION COMPLETE")
            print(f"Pareto Front Size: {len(self.pareto_front)}")
            print(f"Best Solution: C={best_solution.genes[0]:.4f}", end="")
            if self.n_genes > 1:
                print(f", gamma={best_solution.genes[1]:.4f}", end="")
            print(f"\nObjectives: error={best_solution.objectives[0]:.4f}, "
                  f"complexity={best_solution.objectives[1]:.4f}, "
                  f"inv_margin={best_solution.objectives[2]:.4f}")
        
        return {
            'pareto_front': self.pareto_front,
            'best_solution': best_solution,
            'best_C': best_solution.genes[0],
            'best_gamma': best_solution.genes[1] if self.n_genes > 1 else None,
            'best_error': best_solution.objectives[0],
            'history': self.history,
        }


# =============================================================================
# QSVM-NDSGOA PIPELINE
# =============================================================================

class QSVM_NDSGOA:
    """
    Enhanced QSVM with ENSGA optimization (QSVM-NDSGOA).
    
    Pipeline (Algorithm 1):
    1. Encode data using quantum feature map
    2. Compute quantum kernel matrix
    3. Optimize QSVM parameters using ENSGA
    4. Train final QSVM with optimized parameters
    5. Predict and evaluate
    """
    
    def __init__(
        self,
        pop_size: int = 30,
        n_generations: int = 50,
        C_bounds: Tuple[float, float] = (0.01, 100.0),
        gamma_bounds: Tuple[float, float] = (0.001, 10.0),
        optimize_gamma: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize QSVM-NDSGOA.
        
        Args:
            pop_size: ENSGA population size
            n_generations: ENSGA generations
            C_bounds: Bounds for SVM C parameter
            gamma_bounds: Bounds for kernel scaling
            optimize_gamma: Whether to optimize gamma
            verbose: Print progress
        """
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.C_bounds = C_bounds
        self.gamma_bounds = gamma_bounds
        self.optimize_gamma = optimize_gamma
        self.verbose = verbose
        
        # Results
        self.svm = None
        self.best_C = None
        self.best_gamma = None
        self.kernel_matrix = None
        self.pareto_front = None
        self.optimization_results = None
    
    def fit(
        self,
        K_train: np.ndarray,
        y_train: np.ndarray,
        K_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ):
        """
        Train QSVM-NDSGOA.
        
        Args:
            K_train: Precomputed quantum kernel matrix (n_train × n_train)
            y_train: Training labels
            K_val: Validation kernel matrix (optional)
            y_val: Validation labels (optional)
        
        Returns:
            self
        """
        self.kernel_matrix = K_train
        
        if self.verbose:
            print("\n" + "="*60)
            print("QSVM-NDSGOA: ENSGA-Optimized Quantum SVM")
            print("="*60)
        
        # Create objectives
        objectives = QSVMObjectives(
            X_train=None,  # Not needed, using precomputed kernel
            y_train=y_train,
            kernel_matrix=K_train,
        )
        
        # Configure ENSGA
        if self.optimize_gamma:
            bounds = (self.C_bounds, self.gamma_bounds)
        else:
            bounds = (self.C_bounds,)
        
        ensga = ENSGA(
            pop_size=self.pop_size,
            n_generations=self.n_generations,
            bounds=bounds,
            verbose=self.verbose,
        )
        
        # Run optimization
        self.optimization_results = ensga.optimize(objectives)
        
        self.best_C = self.optimization_results['best_C']
        self.best_gamma = self.optimization_results['best_gamma']
        self.pareto_front = self.optimization_results['pareto_front']
        
        # Train final SVM with optimized parameters
        if self.verbose:
            print("\nTraining final SVM with optimized parameters...")
        
        K_scaled = K_train * self.best_gamma if self.best_gamma else K_train
        
        self.svm = SVC(kernel='precomputed', C=self.best_C, probability=True)
        self.svm.fit(K_scaled, y_train)
        
        # Training accuracy
        y_pred = self.svm.predict(K_scaled)
        train_acc = accuracy_score(y_train, y_pred)
        
        if self.verbose:
            print(f"Training Accuracy: {train_acc:.4f}")
        
        return self
    
    def predict(self, K_test: np.ndarray) -> np.ndarray:
        """
        Predict labels for test kernel matrix.
        
        Args:
            K_test: Test kernel matrix (n_test × n_train)
        
        Returns:
            Predicted labels
        """
        K_scaled = K_test * self.best_gamma if self.best_gamma else K_test
        return self.svm.predict(K_scaled)
    
    def predict_proba(self, K_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            K_test: Test kernel matrix (n_test × n_train)
        
        Returns:
            Class probabilities
        """
        K_scaled = K_test * self.best_gamma if self.best_gamma else K_test
        return self.svm.predict_proba(K_scaled)
    
    def get_pareto_solutions(self) -> List[Dict]:
        """Get all Pareto-optimal solutions."""
        solutions = []
        for ind in self.pareto_front:
            solutions.append({
                'C': ind.genes[0],
                'gamma': ind.genes[1] if len(ind.genes) > 1 else None,
                'error': ind.objectives[0],
                'complexity': ind.objectives[1],
                'inverse_margin': ind.objectives[2],
            })
        return solutions


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def train_qsvm_ndsgoa(
    K_train: np.ndarray,
    y_train: np.ndarray,
    K_test: np.ndarray = None,
    y_test: np.ndarray = None,
    pop_size: int = 30,
    n_generations: int = 50,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train and evaluate QSVM with ENSGA optimization.
    
    Args:
        K_train: Training kernel matrix
        y_train: Training labels
        K_test: Test kernel matrix (optional)
        y_test: Test labels (optional)
        pop_size: ENSGA population size
        n_generations: ENSGA generations
        verbose: Print progress
    
    Returns:
        Dict with model, predictions, metrics, pareto_front
    """
    # Train
    model = QSVM_NDSGOA(
        pop_size=pop_size,
        n_generations=n_generations,
        verbose=verbose,
    )
    model.fit(K_train, y_train)
    
    results = {
        'model': model,
        'best_C': model.best_C,
        'best_gamma': model.best_gamma,
        'pareto_solutions': model.get_pareto_solutions(),
        'train_accuracy': accuracy_score(y_train, model.predict(K_train)),
    }
    
    # Evaluate on test set
    if K_test is not None and y_test is not None:
        y_pred = model.predict(K_test)
        y_prob = model.predict_proba(K_test)[:, 1]
        
        results['test_predictions'] = y_pred
        results['test_probabilities'] = y_prob
        results['test_accuracy'] = accuracy_score(y_test, y_pred)
        results['test_f1'] = f1_score(y_test, y_pred)
        
        if verbose:
            print(f"\nTest Accuracy: {results['test_accuracy']:.4f}")
            print(f"Test F1 Score: {results['test_f1']:.4f}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    print("ENSGA Optimizer for QSVM - Demo")
    print("="*60)
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=100,
        n_features=16,
        n_informative=8,
        n_redundant=4,
        n_classes=2,
        random_state=42,
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Simulate quantum kernel (using RBF as proxy)
    from sklearn.metrics.pairwise import rbf_kernel
    
    K_train = rbf_kernel(X_train, X_train, gamma=0.1)
    K_test = rbf_kernel(X_test, X_train, gamma=0.1)
    
    # Train QSVM-NDSGOA
    results = train_qsvm_ndsgoa(
        K_train, y_train,
        K_test, y_test,
        pop_size=20,
        n_generations=30,
        verbose=True,
    )
    
    print("\n" + "="*60)
    print("PARETO FRONT SOLUTIONS")
    print("="*60)
    for i, sol in enumerate(results['pareto_solutions'][:5]):
        print(f"{i+1}. C={sol['C']:.4f}, gamma={sol['gamma']:.4f}, "
              f"error={sol['error']:.4f}, complexity={sol['complexity']:.4f}")

