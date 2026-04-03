"""Supervised learning algorithms for the rice_ml package."""

from .decision_tree import DecisionTree, DecisionTreeClassifier
from .distance_metrics import euclidean_distance, manhattan_distance
from .ensemble_methods import (
    BaggingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
)
from .gradient_descent import GradientDescent1D, GradientDescentND
from .knn import KNNClassifier, KNNRegressor
from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression
from .multilayer_perceptron import MultilayerPerceptron
from .perceptron import Perceptron
from .regression_trees import RegressionTree

__all__ = [
    "LinearRegression",
    "LogisticRegression",
    "KNNClassifier",
    "KNNRegressor",
    "Perceptron",
    "MultilayerPerceptron",
    "DecisionTree",
    "DecisionTreeClassifier",
    "RegressionTree",
    "BaggingClassifier",
    "VotingClassifier",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "GradientDescent1D",
    "GradientDescentND",
    "euclidean_distance",
    "manhattan_distance",
]
