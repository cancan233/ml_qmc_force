import numpy as np
import torch
from ase import Atoms
from ase.io import read
from amptorch.ase_utils import AMPtorch
from amptorch.trainer import AtomsTrainer
from ase.io.trajectory import TrajectoryWriter as tw
from ase.io.trajectory import TrajectoryReader as tr

import numpy as np
import argparse
import sys

images = tr(
    "/gpfs/data/brubenst/chuang25/github_repo/ml_qmc_force/examples/c2/DMC/walker4_widerrange/dmc.traj"
)[6:]

print(len(images))
print(images[0].get_cell())
print(images[0].get_positions())
print(images[0].get_potential_energy())

Gs = {
    "default": {
        "G2": {"etas": np.logspace(np.log10(0.05), np.log10(5.0), num=4), "rs_s": [0],},
        "G4": {"etas": [0.005], "zetas": [1.0, 4.0], "gammas": [1.0, -1.0]},
        "cutoff": 6,
    },
}

config = {
    "model": {
        "get_forces": False,
        "num_layers": 3,
        "num_nodes": 5,
        "batchnorm": False,
    },
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
        "raw_data": images,
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
# trainer.load_pretrained(
#     "/users/chuang25/data/chuang25/github_repo/ml_qmc_force/examples/c2/AMP-DMC/checkpoints/2021-02-01-02-51-59-test"
# )
trainer.train()

predictions = trainer.predict(images)


true_energies = np.array([image.get_potential_energy() for image in images])
pred_energies = np.array(predictions["energy"])
pred_forces = np.array(predictions["forces"])

amp_dmc_traj = tw("c2_amptorch_dmc.traj")
print(len(images))
for i in range(len(images)):
    atoms = images[i]
    amp_dmc_traj.write(atoms, energy=pred_energies[i], forces=pred_forces[i])


print("Energy MSE:", np.mean((true_energies - pred_energies) ** 2) / 2)
print("Energy MAE:", np.mean(np.abs(true_energies - pred_energies)) / 2)
print("Energy RMSE:", np.sqrt(np.mean((true_energies - pred_energies) ** 2)) / 2)
