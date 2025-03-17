"""Tensor class for FourthOrderTensor which allows for custom fields."""

import porepy as pp


class TensorAllowingForCustomFields(pp.FourthOrderTensor):
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
