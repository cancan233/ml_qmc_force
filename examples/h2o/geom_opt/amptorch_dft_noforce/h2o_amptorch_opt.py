import numpy as np

from amptorch.ase_utils import AMPtorch
from amptorch.trainer import AtomsTrainer

from ase.optimize import QuasiNewton, MDMin, BFGS
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin

from ase import units
from ase.build import molecule

from ase.md import MDLogger
from ase.calculators.psi4 import Psi4

import numpy as np
import argparse
import sys

from ase.io.trajectory import TrajectoryReader as tr
from ase.io.trajectory import TrajectoryWriter as tw

np.random.seed(1)

train_images = tr("../../00_data/dft/dft_pyscf_ase_force.traj")

atoms = molecule("H2O")
print(atoms.get_positions())
atoms.set_distance(0, 1, 1.3, fix=0)
atoms.set_angle(1, 0, 2, 120)
# atoms.rattle(seed=1, stdev=0.1)
atoms.cell = np.array(np.diag([15, 15.1, 15.2]))
atoms.rotate(90, "z")
atoms.center()
print(atoms.get_positions())
print(atoms.get_distances(0, [1, 2]))
print(atoms.get_angle(2, 0, 1))

Gs = {
    "default": {
        "G2": {"etas": np.logspace(np.log10(0.05), np.log10(5.0), num=4), "rs_s": [0],},
        "G4": {"etas": [0.005], "zetas": [1.0, 4.0], "gammas": [1.0, -1.0]},
        "cutoff": 6,
    },
}

config = {
    "model": {"get_forces": True, "num_layers": 3, "num_nodes": 5, "batchnorm": False,},
    "optim": {
        "force_coefficient": 0,
        "lr": 1e-2,
        "batch_size": 32,
        "epochs": 2000,
        "loss": "mse",
        "metric": "mae",
        "gpus": 0,
    },
    "dataset": {
        "raw_data": train_images,
        "val_split": 0.1,
        "fp_params": Gs,
        "save_fps": True,
        # feature scaling to be used - normalize or standardize
        # normalize requires a range to be specified
        "scaling": {"type": "normalize", "range": (0, 1)},
    },
    "cmd": {
        "debug": False,
        "run_dir": "./",
        "seed": 1,
        "identifier": "test",
        "verbose": True,
        # Weights and Biases used for logging - an account(free) is required
        "logger": False,
    },
}

trainer = AtomsTrainer(config)
# trainer.train()
trainer.load_pretrained(
    "../../01_train/amptorch_dft_noforce/checkpoints/2022-05-09-22-16-49-test/"
)

calc = AMPtorch(trainer)
atoms.set_calculator(calc)

# # Do a quick relaxation of the cluster
# qn = QuasiNewton(atoms, trajectory="h2o_opt.traj")
# qn = MDMin(atoms, trajectory="h2o_opt.traj")
qn = BFGS(atoms, trajectory="h2o_opt.traj", logfile="h2o_opt.log")
# fmax, steps
qn.run(0.05)
