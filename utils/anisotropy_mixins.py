"""File for anisotropic stiffness tensor mixins. 

Currently, the following tensors are found here:
    * Vertical transverse anisotropic stiffness tensor (for inner domain). Brute force
      version, and thus to be deprecated.
    * Simple anisotropy: Only increase stiffness in one direction. 
    * Vertical transverse anisotropic stiffness tensor: Non-brute force version.
      Constructed generally enough to allow for some flexibility in further
      developments/enhancement of it."""

import numpy as np
import porepy as pp
from utils import inner_domain_cells
from utils.stiffness_tensors import StiffnessTensorInnerVTI


class TransverselyAnisotropicStiffnessTensor:
    """To be deprecated: there is a less brute-force version in the bottom of this
    file."""

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
        inner_domain_center = self.params.get("inner_domain_center", None)
        if width == 0:
            return stiffness_tensor
        else:
            indices = inner_domain_cells(
                self=self,
                sd=subdomain,
                width=width,
                inner_domain_center=inner_domain_center,
            )
            anisotropic_stiffness_values = (
                self.transversely_isotropic_stiffness_tensor()
            )

            stiffness_tensor.values[:, :, indices] = anisotropic_stiffness_values[
                :, :, np.newaxis
            ]
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


class InnerDomainVTIStiffnessTensorMixin:
    """Mixin for a stiffness tensor corresponding to an VTI inner domain, isotropic
    outer domain.

    Instead of using the pp.FourthOrderTensor object (found within PorePy) and then
    overriding certain values, we now use another tensor object specifically tailored
    for a VTI medium. That is, an isotropic medium with an inner VTI one.

    """

    def stiffness_tensor(self, subdomain: pp.Grid) -> StiffnessTensorInnerVTI:
        # Fetch inner domain indices such that we can distribute values of the material
        # parameters in arrays according to cell numbers.
        inner_domain_width = self.params.get("inner_domain_width", 0)
        inner_domain_center = self.params.get("inner_domain_center", None)
        inner_cell_indices = inner_domain_cells(
            self=self,
            sd=subdomain,
            width=inner_domain_width,
            inner_domain_center=inner_domain_center,
        )

        # Preparing basis arrays for inner and outer domains
        inner = np.zeros(subdomain.num_cells)
        inner[inner_cell_indices] = 1

        outer = np.ones(subdomain.num_cells)
        outer = outer - inner

        # Standard material values: These are assigned to the outer domain
        lmbda = self.solid.lame_lambda() * outer
        mu = self.solid.shear_modulus() * outer

        # Anisotropy related values: These are assigned to the inner domain
        anisotropy_constants = self.params["anisotropy_constants"]

        mu_parallel = anisotropy_constants["mu_parallel"] * inner
        mu_orthogonal = anisotropy_constants["mu_orthogonal"] * inner

        volumetric_compr_lambda = (
            anisotropy_constants["volumetric_compr_lambda"] * inner
        )

        lambda_parallel = anisotropy_constants["lambda_parallel"] * inner
        lambda_orthogonal = anisotropy_constants["lambda_orthogonal"] * inner

        # Finally a call to the stiffness tensor object itself
        stiffness_tensor = StiffnessTensorInnerVTI(
            mu=mu,
            lmbda=lmbda,
            mu_parallel=mu_parallel,
            mu_orthogonal=mu_orthogonal,
            lambda_parallel=lambda_parallel,
            lambda_orthogonal=lambda_orthogonal,
            volumetric_compr_lambda=volumetric_compr_lambda,
        )

        return stiffness_tensor
