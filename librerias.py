import numpy as np
import pandas as pd
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# Modelización predictiva (Surrogate Model)
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# --- CORRECCIÓN JMETALPY ---
# Los operadores deben importarse de sus subcarpetas específicas
from jmetal.core.problem import OnTheFlyFloatProblem
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.spea2 import SPEA2
from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII, UniformReferenceDirectionFactory

from jmetal.operator.crossover import SBXCrossover         # Corregido
from jmetal.operator.mutation import PolynomialMutation      # Corregido

from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.solution import get_non_dominated_solutions
# ---------------------------

# Indicadores de calidad
# Nota: Si pymoo falla, asegúrate de tenerlo instalado: pip install pymoo
try:
    from pymoo.indicators.hv import HV
except ImportError:
    # Versiones antiguas de pymoo usaban otra ruta
    from pymoo.performance_indicator.hv import Hypervolume as HV

# Visualización
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import imageio