import sys

import numpy as np
import porepy as pp

sys.path.append("../")

from models import DynamicMomentumBalanceABC
from utils import InnerDomainVTIStiffnessTensorMixin


class AnisotropyModelForTesting(
    InnerDomainVTIStiffnessTensorMixin,
    DynamicMomentumBalanceABC,
): ...
