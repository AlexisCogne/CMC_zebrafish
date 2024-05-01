
import csv
import numpy


def update_muscle_param(animat_options):

    with open('muscle_parameters/muscle_parameters_optimization_FN_15000_ZC_1500_G0_419_gen_99.csv', newline='') as csvfile:
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
            if joint_i == 12:
                mult = 2
                sqrt_mult = numpy.sqrt(mult)
                animat_options["control"]["muscles"][joint_i]["alpha"] *= mult
                animat_options["control"]["muscles"][joint_i]["beta"] *= mult
                animat_options["control"]["muscles"][joint_i]["delta"] *= sqrt_mult
            if joint_i == 13:
                mult = 5
                sqrt_mult = numpy.sqrt(mult)
                animat_options["control"]["muscles"][joint_i]["alpha"] *= mult
                animat_options["control"]["muscles"][joint_i]["beta"] *= mult
                animat_options["control"]["muscles"][joint_i]["delta"] *= sqrt_mult
            if joint_i == 14:
                mult = 20
                sqrt_mult = numpy.sqrt(mult)
                animat_options["control"]["muscles"][joint_i]["alpha"] *= mult
                animat_options["control"]["muscles"][joint_i]["beta"] *= mult
                animat_options["control"]["muscles"][joint_i]["delta"] *= sqrt_mult


def update_drag_param(animat_options):
    for link in animat_options["morphology"]["links"]:
        link["drag_coefficients"][0][1] = 0.3*link["drag_coefficients"][0][1]
        link["drag_coefficients"][0][2] = -0.7

