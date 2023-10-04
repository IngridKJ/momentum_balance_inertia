import porepy as pp

# --------------------------------------------------------
solid_constants_set_1 = pp.SolidConstants(
    {
        "density": 2.0,
        "lame_lambda": 3.0,
        "shear_modulus": 4.0,
    }
)

material_constants_set_1 = {"solid": solid_constants_set_1}
# --------------------------------------------------------

# --------------------------------------------------------
solid_constants_set_2 = pp.SolidConstants(
    {
        "density": 2.0,
        "lame_lambda": 9.0,
        "shear_modulus": 5.0,
    }
)

material_constants_set_2 = {"solid": solid_constants_set_2}
# --------------------------------------------------------
