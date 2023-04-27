from porepy.models.momentum_balance import MomentumBalance
from porepy.applications.md_grids.domains import nd_cube_domain

class ModifiedGeometry:
    def set_domain(self) -> None:
        size = 1/self.units.m
        self._domain = nd_cube_domain(2, size)
    
    def grid_type(self) -> str:
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        mesh_args: dict[str, float] = {"cell_size": 0.1 / self.units.m}
        return mesh_args

class MyMomentumBalance(
    ModifiedGeometry,
    MomentumBalance):
    ...