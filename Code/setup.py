import numpy as np
import matplotlib.pyplot as plt
import logging
import numpy as np
from docplex.mp.model_reader import ModelReader
import matplotlib.pyplot as plt
import numpy as np
from docplex.mp.basic import Expr
from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution
from docplex.util.status import JobSolveStatus
import cplex.callbacks as cpx_cb

# from docplex.mp.callbacks.cb_mixin import ModelCallbackMixin
from docplex.mp.model import Model
from docplex.mp.relax_linear import LinearRelaxer
import os
import sys
