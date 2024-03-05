import porepy as pp
import numpy as np

import sys

sys.path.append("../")

from models import MomentumBalanceABC1
from utils import InnerDomainVTIStiffnessTensorMixin


class AnisotropyModelForTesting(
    InnerDomainVTIStiffnessTensorMixin,
    MomentumBalanceABC1,
): ...
