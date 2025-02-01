from .NeuralNet import NNetWrapper
<<<<<<< HEAD
=======
from .NeuralNet import NeuralNet
>>>>>>> f65eafa0d13d4631a28958d417bfaa503e508930
from .taflNNet import TaflNNet
from .utils import *

# whatever code interpreter you are using might give a warning, but shouldn't matter.
# self note: I'm not sure why, but calling
# from ._azhnefatafl import self_play_function or
# from azhnefatafl._azhnefatafl import self_play_function 
# will fail.  30.01.2025 Keigo

from ._azhnefatafl import self_play_function
