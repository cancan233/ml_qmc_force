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

images = tr(
    "/users/chuang25/data/chuang25/github_repo/ml_qmc_force/examples/c2/dft/widerange/dft.traj"
)[14]

train_images = tr(
    "/gpfs/data/brubenst/chuang25/github_repo/ml_qmc_force/examples/c2/dft/widerange/dft.traj"
)[6:]

atoms = images
model = "amptorch"

print(atoms.get_distances(0, 1))
atoms.set_distance(0, 1, 2)
print(atoms.get_distances(0, 1))
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
        "batch_size": 16,
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
    "/users/chuang25/data/chuang25/github_repo/ml_qmc_force/examples/c2/amptorch_dft/noforce/checkpoints/2022-01-04-10-56-41-test"
)

predictions = trainer.predict(train_images)


# true_energies = np.array([image.get_potential_energy() for image in images])
# pred_energies = np.array(predictions["energy"])

# print("Energy MSE:", np.mean((true_energies - pred_energies) ** 2) / 3)
# print("Energy MAE:", np.mean(np.abs(true_energies - pred_energies)) / 3)
# print("Energy RMSE:", np.sqrt(np.mean((true_energies - pred_energies) ** 2)) / 3)


calc = AMPtorch(trainer)

atoms.set_calculator(calc)
dyn = QuasiNewton(atoms, trajectory="c2_opt.traj")
dyn.run(fmax=0.05)
print(atoms.get_distances(0, 1))
