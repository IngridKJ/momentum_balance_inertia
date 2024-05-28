"""Very preliminary testing, not very nicely done at the moment. But it is a start.

Tests the stiffness tensor that represents inner anisotropic and outer isotropic domain.
Got a weird error message running a simulation (which I for some reason cannot recreate)
that motivated being more proper with testing.

"""

import sys

sys.path.append("../")

import numpy as np
import porepy as pp
from anisotropic_model_for_testing import AnisotropyModelForTesting
from utils import inner_domain_cells


class Model(AnisotropyModelForTesting):
    def nd_rect_domain(self, x, y, z) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y, "zmin": 0, "zmax": z})

        return pp.Domain(box)

    def set_domain(self) -> None:
        x = self.solid.convert_units(5.0, "m")
        y = self.solid.convert_units(5.0, "m")
        z = self.solid.convert_units(5.0, "m")
        self._domain = self.nd_rect_domain(x, y, z)

    def meshing_arguments(self) -> dict:
        cell_size = self.solid.convert_units(1.0, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args


def _build_correct_tensor(anisotropy_constants: dict, model):
    # Hard-coded true outer-isotropic-inner-VTI-tensor. Sort of.
    lam_ort = anisotropy_constants["lambda_orthogonal"]
    lam_par = anisotropy_constants["lambda_parallel"]
    mu_ort = anisotropy_constants["mu_orthogonal"]
    mu_par = anisotropy_constants["mu_parallel"]
    lam = model.solid.lame_lambda()
    mu = model.solid.shear_modulus()

    lambda_mat = np.array(
        [
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
        ]
    )

    lambda_parallel_mat = np.array(
        [
            [1, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    lambda_orthogonal_mat = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )

    mu_parallel_mat = np.array(
        [
            [2, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    mu_orthogonal_mat = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 2],
        ]
    )

    vti_tensor = (
        mu_orthogonal_mat * mu_ort
        + mu_parallel_mat * mu_par
        + lambda_orthogonal_mat * lam_ort
        + lambda_parallel_mat * lam_par
        + lambda_mat * lam
    )

    iso_tensor = mu_orthogonal_mat * mu + mu_parallel_mat * mu + lambda_mat * lam
    return vti_tensor, iso_tensor


def test_iso_vti_tensor():
    # Instantiate model with appropriate parameter values
    anisotropy_constants = {
        "mu_parallel": 20,
        "mu_orthogonal": 20,
        "lambda_parallel": 10,
        "lambda_orthogonal": 5,
        "volumetric_compr_lambda": 25,
    }

    solid_constants = pp.SolidConstants({"lame_lambda": 25, "shear_modulus": 30})
    material_constants = {"solid": solid_constants}

    params = {
        "grid_type": "cartesian",
        "manufactured_solution": "simply_zero",
        "inner_domain_width": 3,
        "anisotropy_constants": anisotropy_constants,
        "material_constants": material_constants,
    }

    model = Model(params)

    # Run prepare simulation
    model.prepare_simulation()

    sd = model.mdg.subdomains(dim=3)[0]
    data = model.mdg.subdomain_data(sd)
    stiffness_tensor = data["parameters"]["mechanics"]["fourth_order_tensor"].values

    num_cells = sd.num_cells
    inner_domain_width = model.params.get("inner_domain_width", 0)
    inner_cell_indices = inner_domain_cells(self=model, sd=sd, width=inner_domain_width)
    vti_tensor, iso_tensor = _build_correct_tensor(
        anisotropy_constants=anisotropy_constants, model=model
    )
    full_tensor = np.tile(iso_tensor, (num_cells, 1, 1))

    # The ordering is very different here. So comparison needs to be done with some
    # care. The cell wise tensor that is constructed in this file is fetched for one
    # cell by the indexing [cell_number, :, :]. The one constructed by the mixin is [:,
    # :, cell_number]. I believe this should be fine tho
    full_tensor[inner_cell_indices, :, :] = vti_tensor

    for i in range(num_cells):
        assert np.all(full_tensor[i, :, :] == stiffness_tensor[:, :, i])
