#!/usr/bin/env python3
"""Run salamander simulation"""

import sys
from dm_control.rl.control import PhysicsError
from util.controller import ZebrafishController
from wave_controller import WaveController
from firing_rate_controller import FiringRateController
from farms_amphibious.control.amphibious import (
    GenericController,
    get_generic_controller,
)
from farms_amphibious.control.kinematics import KinematicsController
from farms_amphibious.data.data import (
    GenericData,
    AmphibiousKinematicsData,
    get_amphibious_data,
)
from farms_amphibious.model.options import GenericOptions
from farms_amphibious.callbacks import setup_callbacks
from farms_sim.simulation import (
    setup_from_clargs,
    run_simulation,
)
from farms_mujoco.sensors.camera import CameraCallback, save_video
from farms_core.simulation.options import Simulator
from typing import Union
import util.update_pars
import numpy as np

from farms_core import pylog
pylog.set_level('warning')


def run_experiment(pars):
    """Main"""
    prepath = "models/zebrafish_v1/"
    args = [
        '--simulator',
        'MUJOCO',
        '--simulation_config', 'models/zebrafish_v1/simulation_zebrafish.yaml',
        '--animat_config', 'models/zebrafish_v1/animat_zebrafish.yaml',
        '--arena_config', 'models/zebrafish_v1/arena_zebrafish.yaml',
        '--log_path', 'output',
        '--profile', 'models/zebrafish_v1/output/simulation.profile',
    ]
    sys.argv += args

    # Setup
    pylog.info('Loading options from clargs')
    (
        clargs,
        animat_options,
        sim_options,
        arena_options,
        simulator,
    ) = setup_from_clargs(animat_options_loader=GenericOptions)

    # modify simulation options according to the user pars
    sim_options["timestep"] = pars.timestep
    sim_options["n_iterations"] = pars.n_iterations
    sim_options["video"] = pars.video_record
    sim_options["headless"] = pars.headless
    sim_options["fast"] = pars.fast
    sim_options["video_speed"] = pars.video_speed
    sim_options["video_fps"] = pars.video_fps
    sim_options["record_path"] = pars.log_path+pars.video_name
    sim_options["camera_id"] = pars.camera_id
    sim_options["show_progress"] = pars.show_progress

    # update muscle and drag parameters
    util.update_pars.update_muscle_param(animat_options)
    util.update_pars.update_drag_param(animat_options)

    # load the correct sdf files depending on the version
    animat_options["sdf"] = prepath+"sdf/zebrafish.sdf"

    if pars.random_spine:
        joints = animat_options["morphology"]["joints"]
        for joint in joints:
            joint["initial"] = [0.2*np.random.randn(), 0]

    # load animat data
    animat_data: Union[GenericData, AmphibiousKinematicsData] = (
        get_amphibious_data(
            animat_options=animat_options,
            simulation_options=sim_options,
        )
    )

    if pars.controller == "firing_rate":
        controller = FiringRateController(
            pars
        )
    elif pars.controller == "sine" or pars.controller == "square":
        controller = WaveController(pars)

    animat_network = ZebrafishController(animat_data, controller)

    # Generic controller (OK)
    animat_controller: Union[GenericController, KinematicsController] = (
        get_generic_controller(
            animat_data=animat_data,
            animat_network=animat_network,
            animat_options=animat_options,
            sim_options=sim_options,
        )
    )

    # Additional engine-specific options
    options = {}
    camera = None
    assert simulator == Simulator.MUJOCO
    if sim_options.video:
        camera = CameraCallback(
            camera_id=sim_options.camera_id,
            timestep=sim_options.timestep,
            n_iterations=sim_options.n_iterations,
            fps=sim_options.video_fps,
            speed=sim_options.video_speed,
            width=sim_options.video_resolution[0],
            height=sim_options.video_resolution[1],
        )
    options['callbacks'] = setup_callbacks(
        animat_options=animat_options,
        arena_options=arena_options,
        camera=camera,
    )

    # Simulation
    pylog.info('Creating simulation environment')
    # handle the exception of physics error
    options["handle_exceptions"] = False

    controller.mujoco_error = False
    try:
        run_simulation(
            animat_data=animat_data,
            animat_options=animat_options,
            animat_controller=animat_controller,
            simulation_options=sim_options,
            arena_options=arena_options,
            simulator=simulator,
            **options,
        )
    except PhysicsError:
        controller.mujoco_error = True

    if sim_options.video:
        save_video(
            camera=camera,
            video_path=f'{sim_options.record_path}',
            iteration=sim_options.n_iterations,
        )

    return animat_data, controller

