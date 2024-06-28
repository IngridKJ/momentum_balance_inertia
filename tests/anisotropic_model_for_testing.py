import sys

import numpy as np
import porepy as pp

sys.path.append("../")

from models import DynamicMomentumBalanceABC2
from utils import InnerDomainVTIStiffnessTensorMixin


class AnisotropyModelForTesting(
    InnerDomainVTIStiffnessTensorMixin,
    DynamicMomentumBalanceABC2,
): ...
