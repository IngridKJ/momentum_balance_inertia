import porepy as pp
import numpy as np

import sys

sys.path.append("../")

from models import DynamicMomentumBalanceABC2
from utils import InnerDomainVTIStiffnessTensorMixin


class AnisotropyModelForTesting(
    InnerDomainVTIStiffnessTensorMixin,
    DynamicMomentumBalanceABC2,
): ...
