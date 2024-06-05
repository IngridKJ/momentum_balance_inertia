import sys

import numpy as np
import porepy as pp
from ABC_model_for_testing import MomentumBalanceABCForTesting

sys.path.append("../")
import run_models.run_linear_model as rlm


def test_energy_decay():
    tf = 1.0
    dt = tf / 100

    time_manager = pp.TimeManager(
        schedule=[0.0, tf],
        dt_init=dt,
        constant_dt=True,
    )

    params = {
        "time_manager": time_manager,
        "grid_type": "cartesian",
        "manufactured_solution": "simply_zero",
    }

    model = MomentumBalanceABCForTesting(params)
    rlm.run_linear_model(model, params)

    for sd in model.mdg.subdomains(dim=model.nd):
        vel_op = model.velocity_time_dep_array([sd]) * model.velocity_time_dep_array(
            [sd]
        )
        vel_op_int = model.volume_integral(integrand=vel_op, grids=[sd], dim=2)
        vel_op_int_val = vel_op_int.value(model.equation_system)
        # Sorry, this is ugly, but, yeah:
        vel_op_int_val_expected = np.array(
            [
                1.43028997e-10,
                1.43028997e-10,
                3.27332489e-10,
                3.80095851e-11,
                3.03219636e-11,
                6.17132981e-10,
                2.23374254e-10,
                5.23681956e-11,
                2.52522052e-11,
                9.23654598e-10,
                6.33958656e-11,
                2.40771403e-10,
                2.84678005e-11,
                1.23951447e-10,
                1.04492207e-12,
                6.05529430e-10,
                5.95134064e-12,
                6.98910244e-10,
                3.61530954e-12,
                6.81284216e-10,
                3.61530954e-12,
                6.81284216e-10,
                5.95134064e-12,
                6.98910244e-10,
                1.04492207e-12,
                6.05529430e-10,
                2.84678005e-11,
                1.23951447e-10,
                6.33958656e-11,
                2.40771403e-10,
                2.52522052e-11,
                9.23654598e-10,
                2.23374254e-10,
                5.23681956e-11,
                3.03219636e-11,
                6.17132981e-10,
                3.27332489e-10,
                3.80095851e-11,
                1.43028997e-10,
                1.43028997e-10,
                3.80095851e-11,
                3.27332489e-10,
                2.54121954e-10,
                2.54121954e-10,
                1.79214560e-10,
                1.42450607e-09,
                2.47602783e-11,
                3.13572021e-10,
                7.17836005e-11,
                1.71094303e-09,
                1.60513376e-11,
                1.95705456e-09,
                5.02554516e-12,
                6.01194507e-11,
                2.37622869e-11,
                4.63232629e-10,
                2.60140213e-12,
                1.40185014e-09,
                1.40307310e-13,
                1.92604496e-09,
                1.40307310e-13,
                1.92604496e-09,
                2.60140213e-12,
                1.40185014e-09,
                2.37622869e-11,
                4.63232629e-10,
                5.02554516e-12,
                6.01194507e-11,
                1.60513376e-11,
                1.95705456e-09,
                7.17836005e-11,
                1.71094303e-09,
                2.47602783e-11,
                3.13572021e-10,
                1.79214560e-10,
                1.42450607e-09,
                2.54121954e-10,
                2.54121954e-10,
                3.80095851e-11,
                3.27332489e-10,
                6.17132981e-10,
                3.03219636e-11,
                1.42450607e-09,
                1.79214560e-10,
                1.55042002e-12,
                1.55042002e-12,
                9.25966062e-10,
                1.00521897e-09,
                1.18258180e-10,
                1.56894806e-10,
                1.36394255e-10,
                2.87825672e-09,
                1.69385506e-11,
                6.41161591e-10,
                5.05929553e-12,
                3.79056220e-10,
                1.76850565e-12,
                1.64993932e-09,
                3.10427342e-13,
                2.10469762e-09,
                3.10427342e-13,
                2.10469762e-09,
                1.76850565e-12,
                1.64993932e-09,
                5.05929553e-12,
                3.79056220e-10,
                1.69385506e-11,
                6.41161591e-10,
                1.36394255e-10,
                2.87825672e-09,
                1.18258180e-10,
                1.56894806e-10,
                9.25966062e-10,
                1.00521897e-09,
                1.55042002e-12,
                1.55042002e-12,
                1.42450607e-09,
                1.79214560e-10,
                6.17132981e-10,
                3.03219636e-11,
                5.23681956e-11,
                2.23374254e-10,
                3.13572021e-10,
                2.47602783e-11,
                1.00521897e-09,
                9.25966062e-10,
                2.38436599e-10,
                2.38436599e-10,
                3.40082240e-10,
                4.80315508e-10,
                1.83948208e-10,
                1.18307816e-09,
                5.11253656e-11,
                1.13731080e-09,
                4.16467880e-11,
                7.78081193e-13,
                4.24177187e-11,
                5.08972077e-10,
                6.52728682e-12,
                1.15490677e-09,
                6.52728682e-12,
                1.15490677e-09,
                4.24177187e-11,
                5.08972077e-10,
                4.16467880e-11,
                7.78081193e-13,
                5.11253656e-11,
                1.13731080e-09,
                1.83948208e-10,
                1.18307816e-09,
                3.40082240e-10,
                4.80315508e-10,
                2.38436599e-10,
                2.38436599e-10,
                1.00521897e-09,
                9.25966062e-10,
                3.13572021e-10,
                2.47602783e-11,
                5.23681956e-11,
                2.23374254e-10,
                9.23654598e-10,
                2.52522052e-11,
                1.71094303e-09,
                7.17836005e-11,
                1.56894806e-10,
                1.18258180e-10,
                4.80315508e-10,
                3.40082240e-10,
                1.81710362e-09,
                1.81710362e-09,
                7.34159324e-10,
                1.56863604e-11,
                3.90351651e-11,
                2.36797350e-09,
                7.92206714e-11,
                5.46296649e-10,
                5.64755879e-11,
                4.54791009e-12,
                1.11394783e-11,
                3.68051947e-11,
                1.11394783e-11,
                3.68051947e-11,
                5.64755879e-11,
                4.54791009e-12,
                7.92206714e-11,
                5.46296649e-10,
                3.90351651e-11,
                2.36797350e-09,
                7.34159324e-10,
                1.56863604e-11,
                1.81710362e-09,
                1.81710362e-09,
                4.80315508e-10,
                3.40082240e-10,
                1.56894806e-10,
                1.18258180e-10,
                1.71094303e-09,
                7.17836005e-11,
                9.23654598e-10,
                2.52522052e-11,
                2.40771403e-10,
                6.33958656e-11,
                1.95705456e-09,
                1.60513376e-11,
                2.87825672e-09,
                1.36394255e-10,
                1.18307816e-09,
                1.83948208e-10,
                1.56863604e-11,
                7.34159324e-10,
                1.56733670e-09,
                1.56733670e-09,
                1.72197896e-09,
                2.52894568e-09,
                3.40113218e-10,
                2.22994250e-09,
                3.11162486e-10,
                5.09836541e-10,
                6.46359567e-10,
                2.62097480e-10,
                6.46359567e-10,
                2.62097480e-10,
                3.11162486e-10,
                5.09836541e-10,
                3.40113218e-10,
                2.22994250e-09,
                1.72197896e-09,
                2.52894568e-09,
                1.56733670e-09,
                1.56733670e-09,
                1.56863604e-11,
                7.34159324e-10,
                1.18307816e-09,
                1.83948208e-10,
                2.87825672e-09,
                1.36394255e-10,
                1.95705456e-09,
                1.60513376e-11,
                2.40771403e-10,
                6.33958656e-11,
                1.23951447e-10,
                2.84678005e-11,
                6.01194507e-11,
                5.02554516e-12,
                6.41161591e-10,
                1.69385506e-11,
                1.13731080e-09,
                5.11253656e-11,
                2.36797350e-09,
                3.90351651e-11,
                2.52894568e-09,
                1.72197896e-09,
                6.92348230e-10,
                6.92348230e-10,
                9.95543725e-15,
                5.13928033e-09,
                2.75572034e-10,
                8.58173858e-10,
                8.51394199e-10,
                7.45710815e-10,
                8.51394199e-10,
                7.45710815e-10,
                2.75572034e-10,
                8.58173858e-10,
                9.95543725e-15,
                5.13928033e-09,
                6.92348230e-10,
                6.92348230e-10,
                2.52894568e-09,
                1.72197896e-09,
                2.36797350e-09,
                3.90351651e-11,
                1.13731080e-09,
                5.11253656e-11,
                6.41161591e-10,
                1.69385506e-11,
                6.01194507e-11,
                5.02554516e-12,
                1.23951447e-10,
                2.84678005e-11,
                6.05529430e-10,
                1.04492207e-12,
                4.63232629e-10,
                2.37622869e-11,
                3.79056220e-10,
                5.05929553e-12,
                7.78081193e-13,
                4.16467880e-11,
                5.46296649e-10,
                7.92206714e-11,
                2.22994250e-09,
                3.40113218e-10,
                5.13928033e-09,
                9.95543725e-15,
                6.56624362e-09,
                6.56624362e-09,
                4.90224755e-09,
                1.02019164e-09,
                3.56681824e-09,
                9.77026318e-10,
                3.56681824e-09,
                9.77026318e-10,
                4.90224755e-09,
                1.02019164e-09,
                6.56624362e-09,
                6.56624362e-09,
                5.13928033e-09,
                9.95543725e-15,
                2.22994250e-09,
                3.40113218e-10,
                5.46296649e-10,
                7.92206714e-11,
                7.78081193e-13,
                4.16467880e-11,
                3.79056220e-10,
                5.05929553e-12,
                4.63232629e-10,
                2.37622869e-11,
                6.05529430e-10,
                1.04492207e-12,
                6.98910244e-10,
                5.95134064e-12,
                1.40185014e-09,
                2.60140213e-12,
                1.64993932e-09,
                1.76850565e-12,
                5.08972077e-10,
                4.24177187e-11,
                4.54791009e-12,
                5.64755879e-11,
                5.09836541e-10,
                3.11162486e-10,
                8.58173858e-10,
                2.75572034e-10,
                1.02019164e-09,
                4.90224755e-09,
                1.70819567e-09,
                1.70819567e-09,
                1.81832464e-09,
                8.45808736e-10,
                1.81832464e-09,
                8.45808736e-10,
                1.70819567e-09,
                1.70819567e-09,
                1.02019164e-09,
                4.90224755e-09,
                8.58173858e-10,
                2.75572034e-10,
                5.09836541e-10,
                3.11162486e-10,
                4.54791009e-12,
                5.64755879e-11,
                5.08972077e-10,
                4.24177187e-11,
                1.64993932e-09,
                1.76850565e-12,
                1.40185014e-09,
                2.60140213e-12,
                6.98910244e-10,
                5.95134064e-12,
                6.81284216e-10,
                3.61530954e-12,
                1.92604496e-09,
                1.40307310e-13,
                2.10469762e-09,
                3.10427342e-13,
                1.15490677e-09,
                6.52728682e-12,
                3.68051947e-11,
                1.11394783e-11,
                2.62097480e-10,
                6.46359567e-10,
                7.45710815e-10,
                8.51394199e-10,
                9.77026318e-10,
                3.56681824e-09,
                8.45808736e-10,
                1.81832464e-09,
                7.68899522e-10,
                7.68899522e-10,
                7.68899522e-10,
                7.68899522e-10,
                8.45808736e-10,
                1.81832464e-09,
                9.77026318e-10,
                3.56681824e-09,
                7.45710815e-10,
                8.51394199e-10,
                2.62097480e-10,
                6.46359567e-10,
                3.68051947e-11,
                1.11394783e-11,
                1.15490677e-09,
                6.52728682e-12,
                2.10469762e-09,
                3.10427342e-13,
                1.92604496e-09,
                1.40307310e-13,
                6.81284216e-10,
                3.61530954e-12,
                6.81284216e-10,
                3.61530954e-12,
                1.92604496e-09,
                1.40307310e-13,
                2.10469762e-09,
                3.10427342e-13,
                1.15490677e-09,
                6.52728682e-12,
                3.68051947e-11,
                1.11394783e-11,
                2.62097480e-10,
                6.46359567e-10,
                7.45710815e-10,
                8.51394199e-10,
                9.77026318e-10,
                3.56681824e-09,
                8.45808736e-10,
                1.81832464e-09,
                7.68899522e-10,
                7.68899522e-10,
                7.68899522e-10,
                7.68899522e-10,
                8.45808736e-10,
                1.81832464e-09,
                9.77026318e-10,
                3.56681824e-09,
                7.45710815e-10,
                8.51394199e-10,
                2.62097480e-10,
                6.46359567e-10,
                3.68051947e-11,
                1.11394783e-11,
                1.15490677e-09,
                6.52728682e-12,
                2.10469762e-09,
                3.10427342e-13,
                1.92604496e-09,
                1.40307310e-13,
                6.81284216e-10,
                3.61530954e-12,
                6.98910244e-10,
                5.95134064e-12,
                1.40185014e-09,
                2.60140213e-12,
                1.64993932e-09,
                1.76850565e-12,
                5.08972077e-10,
                4.24177187e-11,
                4.54791009e-12,
                5.64755879e-11,
                5.09836541e-10,
                3.11162486e-10,
                8.58173858e-10,
                2.75572034e-10,
                1.02019164e-09,
                4.90224755e-09,
                1.70819567e-09,
                1.70819567e-09,
                1.81832464e-09,
                8.45808736e-10,
                1.81832464e-09,
                8.45808736e-10,
                1.70819567e-09,
                1.70819567e-09,
                1.02019164e-09,
                4.90224755e-09,
                8.58173858e-10,
                2.75572034e-10,
                5.09836541e-10,
                3.11162486e-10,
                4.54791009e-12,
                5.64755879e-11,
                5.08972077e-10,
                4.24177187e-11,
                1.64993932e-09,
                1.76850565e-12,
                1.40185014e-09,
                2.60140213e-12,
                6.98910244e-10,
                5.95134064e-12,
                6.05529430e-10,
                1.04492207e-12,
                4.63232629e-10,
                2.37622869e-11,
                3.79056220e-10,
                5.05929553e-12,
                7.78081193e-13,
                4.16467880e-11,
                5.46296649e-10,
                7.92206714e-11,
                2.22994250e-09,
                3.40113218e-10,
                5.13928033e-09,
                9.95543725e-15,
                6.56624362e-09,
                6.56624362e-09,
                4.90224755e-09,
                1.02019164e-09,
                3.56681824e-09,
                9.77026318e-10,
                3.56681824e-09,
                9.77026318e-10,
                4.90224755e-09,
                1.02019164e-09,
                6.56624362e-09,
                6.56624362e-09,
                5.13928033e-09,
                9.95543725e-15,
                2.22994250e-09,
                3.40113218e-10,
                5.46296649e-10,
                7.92206714e-11,
                7.78081193e-13,
                4.16467880e-11,
                3.79056220e-10,
                5.05929553e-12,
                4.63232629e-10,
                2.37622869e-11,
                6.05529430e-10,
                1.04492207e-12,
                1.23951447e-10,
                2.84678005e-11,
                6.01194507e-11,
                5.02554516e-12,
                6.41161591e-10,
                1.69385506e-11,
                1.13731080e-09,
                5.11253656e-11,
                2.36797350e-09,
                3.90351651e-11,
                2.52894568e-09,
                1.72197896e-09,
                6.92348230e-10,
                6.92348230e-10,
                9.95543725e-15,
                5.13928033e-09,
                2.75572034e-10,
                8.58173858e-10,
                8.51394199e-10,
                7.45710815e-10,
                8.51394199e-10,
                7.45710815e-10,
                2.75572034e-10,
                8.58173858e-10,
                9.95543725e-15,
                5.13928033e-09,
                6.92348230e-10,
                6.92348230e-10,
                2.52894568e-09,
                1.72197896e-09,
                2.36797350e-09,
                3.90351651e-11,
                1.13731080e-09,
                5.11253656e-11,
                6.41161591e-10,
                1.69385506e-11,
                6.01194507e-11,
                5.02554516e-12,
                1.23951447e-10,
                2.84678005e-11,
                2.40771403e-10,
                6.33958656e-11,
                1.95705456e-09,
                1.60513376e-11,
                2.87825672e-09,
                1.36394255e-10,
                1.18307816e-09,
                1.83948208e-10,
                1.56863604e-11,
                7.34159324e-10,
                1.56733670e-09,
                1.56733670e-09,
                1.72197896e-09,
                2.52894568e-09,
                3.40113218e-10,
                2.22994250e-09,
                3.11162486e-10,
                5.09836541e-10,
                6.46359567e-10,
                2.62097480e-10,
                6.46359567e-10,
                2.62097480e-10,
                3.11162486e-10,
                5.09836541e-10,
                3.40113218e-10,
                2.22994250e-09,
                1.72197896e-09,
                2.52894568e-09,
                1.56733670e-09,
                1.56733670e-09,
                1.56863604e-11,
                7.34159324e-10,
                1.18307816e-09,
                1.83948208e-10,
                2.87825672e-09,
                1.36394255e-10,
                1.95705456e-09,
                1.60513376e-11,
                2.40771403e-10,
                6.33958656e-11,
                9.23654598e-10,
                2.52522052e-11,
                1.71094303e-09,
                7.17836005e-11,
                1.56894806e-10,
                1.18258180e-10,
                4.80315508e-10,
                3.40082240e-10,
                1.81710362e-09,
                1.81710362e-09,
                7.34159324e-10,
                1.56863604e-11,
                3.90351651e-11,
                2.36797350e-09,
                7.92206714e-11,
                5.46296649e-10,
                5.64755879e-11,
                4.54791009e-12,
                1.11394783e-11,
                3.68051947e-11,
                1.11394783e-11,
                3.68051947e-11,
                5.64755879e-11,
                4.54791009e-12,
                7.92206714e-11,
                5.46296649e-10,
                3.90351651e-11,
                2.36797350e-09,
                7.34159324e-10,
                1.56863604e-11,
                1.81710362e-09,
                1.81710362e-09,
                4.80315508e-10,
                3.40082240e-10,
                1.56894806e-10,
                1.18258180e-10,
                1.71094303e-09,
                7.17836005e-11,
                9.23654598e-10,
                2.52522052e-11,
                5.23681956e-11,
                2.23374254e-10,
                3.13572021e-10,
                2.47602783e-11,
                1.00521897e-09,
                9.25966062e-10,
                2.38436599e-10,
                2.38436599e-10,
                3.40082240e-10,
                4.80315508e-10,
                1.83948208e-10,
                1.18307816e-09,
                5.11253656e-11,
                1.13731080e-09,
                4.16467880e-11,
                7.78081193e-13,
                4.24177187e-11,
                5.08972077e-10,
                6.52728682e-12,
                1.15490677e-09,
                6.52728682e-12,
                1.15490677e-09,
                4.24177187e-11,
                5.08972077e-10,
                4.16467880e-11,
                7.78081193e-13,
                5.11253656e-11,
                1.13731080e-09,
                1.83948208e-10,
                1.18307816e-09,
                3.40082240e-10,
                4.80315508e-10,
                2.38436599e-10,
                2.38436599e-10,
                1.00521897e-09,
                9.25966062e-10,
                3.13572021e-10,
                2.47602783e-11,
                5.23681956e-11,
                2.23374254e-10,
                6.17132981e-10,
                3.03219636e-11,
                1.42450607e-09,
                1.79214560e-10,
                1.55042002e-12,
                1.55042002e-12,
                9.25966062e-10,
                1.00521897e-09,
                1.18258180e-10,
                1.56894806e-10,
                1.36394255e-10,
                2.87825672e-09,
                1.69385506e-11,
                6.41161591e-10,
                5.05929553e-12,
                3.79056220e-10,
                1.76850565e-12,
                1.64993932e-09,
                3.10427342e-13,
                2.10469762e-09,
                3.10427342e-13,
                2.10469762e-09,
                1.76850565e-12,
                1.64993932e-09,
                5.05929553e-12,
                3.79056220e-10,
                1.69385506e-11,
                6.41161591e-10,
                1.36394255e-10,
                2.87825672e-09,
                1.18258180e-10,
                1.56894806e-10,
                9.25966062e-10,
                1.00521897e-09,
                1.55042002e-12,
                1.55042002e-12,
                1.42450607e-09,
                1.79214560e-10,
                6.17132981e-10,
                3.03219636e-11,
                3.80095851e-11,
                3.27332489e-10,
                2.54121954e-10,
                2.54121954e-10,
                1.79214560e-10,
                1.42450607e-09,
                2.47602783e-11,
                3.13572021e-10,
                7.17836005e-11,
                1.71094303e-09,
                1.60513376e-11,
                1.95705456e-09,
                5.02554516e-12,
                6.01194507e-11,
                2.37622869e-11,
                4.63232629e-10,
                2.60140213e-12,
                1.40185014e-09,
                1.40307310e-13,
                1.92604496e-09,
                1.40307310e-13,
                1.92604496e-09,
                2.60140213e-12,
                1.40185014e-09,
                2.37622869e-11,
                4.63232629e-10,
                5.02554516e-12,
                6.01194507e-11,
                1.60513376e-11,
                1.95705456e-09,
                7.17836005e-11,
                1.71094303e-09,
                2.47602783e-11,
                3.13572021e-10,
                1.79214560e-10,
                1.42450607e-09,
                2.54121954e-10,
                2.54121954e-10,
                3.80095851e-11,
                3.27332489e-10,
                1.43028997e-10,
                1.43028997e-10,
                3.27332489e-10,
                3.80095851e-11,
                3.03219636e-11,
                6.17132981e-10,
                2.23374254e-10,
                5.23681956e-11,
                2.52522052e-11,
                9.23654598e-10,
                6.33958656e-11,
                2.40771403e-10,
                2.84678005e-11,
                1.23951447e-10,
                1.04492207e-12,
                6.05529430e-10,
                5.95134064e-12,
                6.98910244e-10,
                3.61530954e-12,
                6.81284216e-10,
                3.61530954e-12,
                6.81284216e-10,
                5.95134064e-12,
                6.98910244e-10,
                1.04492207e-12,
                6.05529430e-10,
                2.84678005e-11,
                1.23951447e-10,
                6.33958656e-11,
                2.40771403e-10,
                2.52522052e-11,
                9.23654598e-10,
                2.23374254e-10,
                5.23681956e-11,
                3.03219636e-11,
                6.17132981e-10,
                3.27332489e-10,
                3.80095851e-11,
                1.43028997e-10,
                1.43028997e-10,
            ]
        )
        assert np.isclose(np.linalg.norm(vel_op_int_val), 3.859854428354451e-08)
        assert np.allclose(vel_op_int_val_expected, vel_op_int_val)
