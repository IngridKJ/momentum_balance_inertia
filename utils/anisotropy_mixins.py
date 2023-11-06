"""File for anisotropic tiffness tensor mixins"""

import porepy as pp
import numpy as np
from utils import inner_domain_cells


class TransverselyAnisotropicStiffnessTensor:
    def stiffness_tensor(self, subdomain: pp.Grid) -> pp.FourthOrderTensor:
        """Stiffness tensor [Pa].

        Modified to represent a transversely isotropic stiffness tensor. It is
        (for now) done in a rather brute force way. All cells corresponding to the
        "inner domain" (see the function inner_domain_cells in utils) will have their
        stiffness tensor values wiped. A newly created 9x9 matrix corresponding to the
        values of a transversely isotropic medium is assigned to the wiped part.

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
            anisotropic_stiffness_values = (
                self.transversely_isotropic_stiffness_tensor()
            )
            for cell_index in indices:
                stiffness_tensor.values[:, :, cell_index] *= 0
                stiffness_tensor.values[
                    :, :, cell_index
                ] += anisotropic_stiffness_values

        return stiffness_tensor

    def transversely_isotropic_stiffness_tensor(self) -> np.ndarray:
        """Matrices representing the anisotropic stiffness tensor contribution.

        Stiffness tensor for transverse isotropy in z-direction is created here. The
        anisotropic parameter values are found within
        (self.params["anisotropy_constants"]). If no values are assigned to this
        dicrionary, a zero contribution is returned. That is, the stiffness tensor is
        again representing an isotropic medium.

        This method is used by stiffness_tensor as a utility method.

        Returns:
            A 9x9 matrix with the anisotropic stiffness tensor values.

        """
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
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
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

        anisotropy_constants = self.params.get("anisotropy_constants", None)
        if anisotropy_constants is None:
            return np.zeros((9, 9))
        else:
            mu_parallel = anisotropy_constants["mu_parallel"]
            mu_orthogonal = anisotropy_constants["mu_orthogonal"]

            volumetric_compr_lambda = self.solid.lame_lambda()

            lambda_parallel = anisotropy_constants["lambda_parallel"]
            lambda_orthogonal = anisotropy_constants["lambda_orthogonal"]

            values = (
                mu_parallel * mu_parallel_mat
                + mu_orthogonal * mu_orthogonal_mat
                + lambda_orthogonal * lambda_orthogonal_mat
                + lambda_parallel * lambda_parallel_mat
                + volumetric_compr_lambda * lambda_mat
            )
            return values


class SimpleAnisotropy:
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
