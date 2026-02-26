# TONY/IRF/__init__.py
from .IRF import *

try:
    from .IRF_mlx import *  # solo Apple Silicon
except ImportError:
    pass

