"""File containing the classes required for a fourth order stiffness tensor. 

Analogous to the tensors.py module within PorePy, but this one allows for custom fields.
That is, instead of only mu and lambda, one can pass additional fields (with their
corresponding 9x9 matrices). Something similar to this class is on its way into PorePy,
but it is not there yet."""

import numpy as np
import porepy as pp

class TensorAllowingForCustomFields(pp.FourthOrderTensor):
    def __init__(
        self,
        mu: np.ndarray,
        lmbda: np.ndarray,
        other_fields: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    ):
        """Constructor for fourth order tensor on Lame-parameter form.

        Implementation somewhat mirrors that found in the branch restrict_tensor (by
        Eirik).
        """

        # Save lmbda and mu, can be useful to have in some cases
        self.lmbda = lmbda
        """Nc array of first Lamé parameter."""

        self.mu = mu
        """Nc array of shear modulus (second Lamé parameter)."""

        # Basis for the contributions of mu and lambda is hard-coded
        mu_mat = np.array(
            [
                [2, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 2],
            ]
        )
        lmbda_mat = np.array(
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

        # Expand dimensions to prepare for cell-wise representation.
        mu_mat = mu_mat[:, :, np.newaxis]
        lmbda_mat = lmbda_mat[:, :, np.newaxis]

        # List of constitutive parameters
        self._constitutive_parameters = ["mu", "lmbda"]

        c = mu_mat * mu + lmbda_mat * lmbda

        # Store the other fields. This is needed for the copy method.
        self._other_matrices = {}

        for key, (mat, field) in other_fields.items():
            c += mat[:, :, np.newaxis] * field
            setattr(self, key, field)
            self._other_matrices[key] = mat
            self._constitutive_parameters.append(key)

        self.values = c
        """Values of the stiffness tensor as a (3^2, 3^2, Nc) array."""

    def copy(self):
        """Define a deep copy of the tensor.

        Returns:
            FourthOrderTensor: New tensor with identical fields, but separate arrays (in
                the memory sense).

        """
        extra_params = {}
        for key, mat in self._other_matrices.items():
            extra_params[key] = (mat, getattr(self, key).copy())

        C = TensorAllowingForCustomFields(
            mu=self.mu.copy(), lmbda=self.lmbda.copy(), other_fields=extra_params
        )
        C.values = self.values.copy()
        return C

    @staticmethod
    def constitutive_parameters(self):
        return [
            "mu",
            "lmbda",
            "mu_parallel",
            "mu_orthogonal",
            "lambda_parallel",
            "lambda_orthogonal",
            "volumetric_compr_lambda",
        ]
