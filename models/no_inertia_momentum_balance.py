"""Static momentum balance model. 

Relies on the momentum balance model class found within PorePy. The only adaptation is to the geometry.

"""

from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.models.momentum_balance import MomentumBalance


class ModifiedGeometry:
    def set_domain(self) -> None:
        size = self.solid.convert_units(1.0, "m")
        self._domain = nd_cube_domain(2, size)

    def grid_type(self) -> str:
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        cell_size = self.solid.convert_units(0.1, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args


class MomentumBalanceModified(ModifiedGeometry, MomentumBalance): ...
