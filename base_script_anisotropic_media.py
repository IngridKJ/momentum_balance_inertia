import porepy as pp
import numpy as np

from models.absorbing_boundary_conditions import MomentumBalanceABC

from utils import inner_domain_cells


class AnisotropicStiffnessTensor:
    def stiffness_tensor(self, subdomain: pp.Grid) -> pp.FourthOrderTensor:
        """Stiffness tensor [Pa].

        Parameters:
            subdomain: Subdomain where the stiffness tensor is defined.

        Returns:
            Cell-wise stiffness tensor in SI units.

        """
        lmbda = self.solid.lame_lambda() * np.ones(subdomain.num_cells)
        mu = self.solid.shear_modulus() * np.ones(subdomain.num_cells)
        stiffness_tensor = pp.FourthOrderTensor(mu, lmbda)

        width = self.params.get("inner_domain_width", 0)
        if width == 0:
            return stiffness_tensor
        else:
            indices = inner_domain_cells(self=self, sd=subdomain, width=width)
            anisotropic_stiffness_values = self.construct_anisotropic_contribution()
            for cell_index in indices:
                stiffness_tensor.values[
                    :, :, cell_index
                ] += anisotropic_stiffness_values

        return stiffness_tensor

    def construct_anisotropic_contribution(self) -> np.ndarray:
        """Matrices representing the anisotropic stiffness tensor contribution.

        The matrices found here are a simple form of anisotropy. They increase the
        stiffness in one (or more) directions parallel with the coordinate axes. Which
        directions that have increased stiffness depends on the values of lmbda1, lmbda2
        and lmbda3 in the params dictionary (self.params["anisotropy_constants"]). If no
        values are assigned, a zero contribution is returned. That is, the stiffness
        tensor is again representing an isotropic medium.

        This method is used by stiffness_tensor as a utility method.

        Returns:
            Sum of all anisotropic contributions in the shape of a 9 by 9 matrix.

        """
        lmbda1_mat = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        lmbda2_mat = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        lmbda3_mat = np.array(
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
        anisotropy_constants = self.params.get("anisotropy_constants", None)
        if anisotropy_constants is None:
            return np.zeros((9, 9))
        else:
            lmbda1 = anisotropy_constants["lmbda1"]
            lmbda2 = anisotropy_constants["lmbda2"]
            lmbda3 = anisotropy_constants["lmbda3"]

            values = lmbda1_mat * lmbda1 + lmbda2_mat * lmbda2 + lmbda3_mat * lmbda3
            return values


class AnisotropicBaseModel(
    AnisotropicStiffnessTensor,
    MomentumBalanceABC,
):
    ...
