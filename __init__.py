import os
import time
import random
import math
import urllib.request
import gzip
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from typing import List, Dict, Literal, Callable, Final
from tqdm import tqdm
from itertools import product
from copy import deepcopy
from tabulate import tabulate
plt.rcParams['font.size'] = 14
sns.set_theme()