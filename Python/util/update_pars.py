
import csv


def update_muscle_param(animat_options, **kwargs):

    with open('muscle_parameters/muscle_parameters_optimization_FN_20000_ZC_1000_G0_419_gen_99.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            joint_i = int(row[''])
            animat_options["control"]["muscles"][joint_i]["alpha"] = float(
                row['alpha'])
            animat_options["control"]["muscles"][joint_i]["beta"] = float(
                row['beta'])
            animat_options["control"]["muscles"][joint_i]["gamma"] = 1
            animat_options["control"]["muscles"][joint_i]["delta"] = float(
                row['delta'])
            if joint_i > 13:
                # stiffening the tail joint(s)
                animat_options["control"]["muscles"][joint_i]["beta"] *= 10


def update_drag_param(animat_options, **kwargs):
    for link in animat_options["morphology"]["links"]:
        # # oroginal jon's parameters
        # link["drag_coefficients"][0][0]=-0.001
        # link["drag_coefficients"][0][1]=-0.1 # reduce to make stable
        # link["drag_coefficients"][0][2]=-0.3

        factor = 0.1  # factor for the drags
        link["drag_coefficients"][0][0] = factor * \
            link["drag_coefficients"][0][0]
        link["drag_coefficients"][0][1] = factor * \
            link["drag_coefficients"][0][1]
        # factor*link["drag_coefficients"][0][2] # increase z drag to avoid
        # rolling
        link["drag_coefficients"][0][2] = -0.1

