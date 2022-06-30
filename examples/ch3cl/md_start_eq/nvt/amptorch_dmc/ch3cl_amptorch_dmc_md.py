import numpy as np
import torch
from ase import Atoms
from ase.io import read
from amptorch.ase_utils import AMPtorch
from amptorch.trainer import AtomsTrainer
from ase.optimize import QuasiNewton
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin

from ase import units
from ase.build import molecule
from amp import Amp
from ase.md import MDLogger
from ase.calculators.psi4 import Psi4

import numpy as np
import argparse
import sys

from ase.io.trajectory import TrajectoryReader as tr
from ase.io.trajectory import TrajectoryWriter as tw

train_images = tr(
    "/users/chuang25/data/chuang25/github_repo/ml_qmc_force/examples/ch3cl/dmc/pyscf/test_dmc.traj"
)

simtype = "nvt"
timestep = 0.1
time = 20000
model = "amptorch"

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
        "lr": 1e-3,
        "batch_size": 64,
        "epochs": 1000,
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
# 10-48 is the checkpoint trained with 2000 epochs
trainer.load_pretrained(
    "/users/chuang25/data/chuang25/github_repo/ml_qmc_force/examples/ch3cl/amptorch_dmc/pyscf/train_test03/checkpoints/2022-05-10-22-17-18-test"
)

# predictions = trainer.predict(train_images)

calc = AMPtorch(trainer)

atoms = tr(
    "/users/chuang25/data/chuang25/github_repo/ml_qmc_force/examples/ch3cl/MD_start_eq/dft_pyscf_ase/nvt/md_nvt_dft.traj"
)[0]
atoms.set_calculator(calc)

MaxwellBoltzmannDistribution(atoms, 300 * units.kB)
Stationary(atoms)
ZeroRotation(atoms)

if simtype == "nve":
    dyn = VelocityVerlet(
        atoms,
        timestep * units.fs,
        trajectory="md_{}_{}.traj".format(simtype, model),
        logfile="md_{}_{}.log".format(simtype, model),
        loginterval=10,
    )
elif simtype == "nvt":
    dyn = Langevin(
        atoms,
        timestep * units.fs,
        300 * units.kB,
        friction=2.5,
        trajectory="md_{}_{}.traj".format(simtype, model),
        logfile="md_{}_{}.log".format(simtype, model),
        loginterval=10,
    )

dyn.run(time)
