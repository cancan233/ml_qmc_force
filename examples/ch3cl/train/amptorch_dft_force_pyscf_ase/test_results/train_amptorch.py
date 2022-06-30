from re import T
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

np.random.seed(1)

images = tr(
    "/Users/cancan/Dropbox (Brown)/000_Research/ML_DMC/examples/ch3cl/data/dft/dft_pyscf_ase_force.traj"
)

print(images[0].get_cell())
print(images[0].get_positions())
print(images[0].get_potential_energy())
print(images[0].get_forces())

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
        "force_coefficient": 0.04,
        "lr": 1e-3,
        "batch_size": 32,
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
trainer.load_pretrained(
    "../2022-04-28-23-52-06-test"
)

train_images = images
predictions = trainer.predict(train_images)

true_energies = np.array([image.get_potential_energy() for image in train_images])
pred_energies = np.array(predictions["energy"])
pred_forces = np.array(predictions["forces"])

print("TRAIN Energy MSE:", np.mean((true_energies - pred_energies) ** 2) / 3)
print("TRAIN Energy MAE:", np.mean(np.abs(true_energies - pred_energies)) / 3)
print("TRAIN Energy RMSE:", np.sqrt(np.mean((true_energies - pred_energies) ** 2)) / 3)
amp_dmc_traj = tw("ch3cl_amptorch_dft_train.traj")
print(len(train_images))
for i in range(len(train_images)):
    atoms = train_images[i]
    atoms.info["true_energy"] = true_energies[i]
    amp_dmc_traj.write(atoms, energy=pred_energies[i], forces=pred_forces[i])

test_images = tr("/Users/cancan/Dropbox (Brown)/000_Research/ML_DMC/examples/ch3cl/data/test_dft_with_force.traj")
predictions = trainer.predict(test_images)

true_energies = np.array([image.get_potential_energy() for image in test_images])
pred_energies = np.array(predictions["energy"])
pred_forces = np.array(predictions["forces"])

print("TEST Energy MSE:", np.mean((true_energies - pred_energies) ** 2) / 3)
print("TEST Energy MAE:", np.mean(np.abs(true_energies - pred_energies)) / 3)
print("TEST Energy RMSE:", np.sqrt(np.mean((true_energies - pred_energies) ** 2)) / 3)
amp_dmc_traj = tw("ch3cl_amptorch_dft_test.traj")
print(len(test_images))
for i in range(len(test_images)):
    atoms = test_images[i]
    atoms.info["true_energy"] = true_energies[i]
    amp_dmc_traj.write(atoms, energy=pred_energies[i], forces=pred_forces[i])
