import porepy as pp
import numpy as np

from models import MomentumBalanceABC

from utils import InnerDomainVTIStiffnessTensorMixin


class AnisotropyModelForTesting(
    InnerDomainVTIStiffnessTensorMixin,
    MomentumBalanceABC,
):
    ...
