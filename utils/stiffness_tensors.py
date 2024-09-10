"""File containing classes for a fourth order stiffness tensor. 

Analogous to the tensors.py module within PorePy, but this one provides a tensor
corresponding to an anisotropic medium."""

import numpy as np


class StiffnessTensorInnerVTI(object):
    """Cell-wise representation of fourth order tensor.

    The tensor is represented by a (3^2, 3^2 ,Nc)-array, where Nc denotes the number of cells, i.e. the tensor values are stored discretely.

    See pp.FourthOrderTensor in PorePy for more extensive documentation.

    The only constructor available (here, in this file) for the moment is based on five
    independent material parameters. It should also be noted that the construction of
    the stiffness tensor has one intended use as of now: We create a tensor that
    represents an outer isotropic domain and an inner vertically transverse isotropic
    inner domain. Therefore you will see 7 material parameter attributes instead of only
    5. 2 are for the isotropic domain, 5 are for the anisotropic one.

    Attributes:
        values (numpy.ndarray): dimensions (3^2, 3^2, nc), cell-wise
            representation of the stiffness matrix.
        lmbda (np.ndarray): Nc array of first Lame parameter.
        mu (np.ndarray): Nc array of second Lame parameter.
        mu_parallel (np.ndarray): Nc array of transverse shear parameter.
        mu_orthogonal (np.ndarray): Nc array of transverse-to-perpendicular shear
            parameter.
        lambda_parallell (np.ndarray): Transverse compressive stress parameter.
        lambda_orthogonal (np.ndarray): Perpendicular compressive stress parameter.
        volumetric_compr_lambda (np.ndarray): Volumetric compressive stress parameter.

    """

    def __init__(
        self,
        mu: np.ndarray,
        lmbda: np.ndarray,
        mu_parallel: np.ndarray,
        mu_orthogonal: np.ndarray,
        lambda_parallel: np.ndarray,
        lambda_orthogonal: np.ndarray,
        volumetric_compr_lambda: np.ndarray,
    ):
        """Constructor for the fourth order tensor.

        Parameters:
            lmbda: Nc array of first Lame parameter.
            mu: Nc array of second Lame parameter.
            mu_parallel: Nc array of transverse shear parameter.
            mu_orthogonal: Nc array of transverse-to-perpendicular shear parameter.
            lambda_parallell: Transverse compressive stress parameter.
            lambda_orthogonal: Perpendicular compressive stress parameter.
            volumetric_compr_lambda: Volumetric compressive stress parameter.
        """
        # Creating attributes of the values, as this might come in handy later
        self.lmbda = lmbda
        self.mu = mu

        self.mu_parallel = mu_parallel
        self.mu_orthogonal = mu_orthogonal

        self.volumetric_compr_lambda = volumetric_compr_lambda
        self.lambda_parallel = lambda_parallel
        self.lambda_orthogonal = lambda_orthogonal

        # Basis for mu and lambda contribution for tensor build
        (
            l_mat,
            l_par_mat,
            l_ort_mat,
            m_par_mat,
            m_ort_mat,
        ) = self.hardcoded_mu_lam_basis()

        # Matrices for the isotropic tensor build
        lmbda_mat = l_mat
        mu_mat = m_par_mat + m_ort_mat

        # Adding axis to isotropic related matrices
        mu_mat = mu_mat[:, :, np.newaxis]
        lmbda_mat = lmbda_mat[:, :, np.newaxis]

        # Adding axis to anisotropy related matrices
        l_mat = l_mat[:, :, np.newaxis]
        l_par_mat = l_par_mat[:, :, np.newaxis]
        l_ort_mat = l_ort_mat[:, :, np.newaxis]
        m_par_mat = m_par_mat[:, :, np.newaxis]
        m_ort_mat = m_ort_mat[:, :, np.newaxis]

        c = (
            # Isotropic outer domain
            mu_mat * mu
            + lmbda_mat * lmbda
            # VTI inner domain
            + l_mat * volumetric_compr_lambda
            + l_par_mat * lambda_parallel
            + l_ort_mat * lambda_orthogonal
            + m_par_mat * mu_parallel
            + m_ort_mat * mu_orthogonal
        )

        self.values = c

    def copy(self) -> "StiffnessTensorInnerVTI":
        """
        Define a deep copy of the tensor.

        Returns:
            StiffnessTensorInnerVTI: New tensor with identical fields, but separate
                arrays (in the memory sense).
        """
        C = StiffnessTensorInnerVTI(
            mu=self.mu,
            lmbda=self.lmbda,
            mu_parallel=self.mu_parallel,
            mu_orthogonal=self.mu_orthogonal,
            lambda_parallel=self.lambda_parallel,
            lambda_orthogonal=self.lambda_orthogonal,
            volumetric_compr_lambda=self.volumetric_compr_lambda,
        )
        C.values = self.values.copy()
        return C

    def hardcoded_mu_lam_basis(self) -> tuple:
        """Basis for the contributions of mu and lambda in a VTI stiffness tensor.

        This method contains the five matrices needed for creating a vertical
        transversely isotropic stiffness tensor.

        Returns:
            A tuple of all the matrices in the following order:
                lambda_mat,
                lambda_parallel_mat,
                lambda_orthogonal_mat,
                mu_parallel_mat,
                mu_orthogonal_mat.

        Additional notes:
            lambda_mat is the basis related to the volumetric compressive stress
            parameter.

            The two mu-matrices, namely mu_parallel_mat and mu_orthogonal_mat, are the
            bases related to the transverse and transverse-to-perpendicular shear
            parameter, respectively.

            The two remaining matrices, lambda_parallel_mat and lambda_orthogonal_mat,
            are the bases related to the transverse compressive stress parameter and the
            perpendicular compressive stress parameter, respectively.

            For an isotropic media, lambda_mat is kept as is, and the two mu-matrices
            are summed up (and provided the same values for orthogonal and parallel mu).
            The remaining lambda matrices contribute nothing, as the transverse
            compressive stress parameter and perpendicular compressive stress parameter
            are then set to zero.

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
        return (
            lambda_mat,
            lambda_parallel_mat,
            lambda_orthogonal_mat,
            mu_parallel_mat,
            mu_orthogonal_mat,
        )
