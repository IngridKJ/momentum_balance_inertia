import porepy as pp
import numpy as np

import sys

sys.path.append("../")

from models import MomentumBalanceABC
from utils import InnerDomainVTIStiffnessTensorMixin


class AnisotropyModelForTesting(
    InnerDomainVTIStiffnessTensorMixin,
    MomentumBalanceABC,
):
    ...
