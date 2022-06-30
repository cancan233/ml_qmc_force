import numpy as np
import torch
from ase import Atoms
from ase.io import read
from ase.io.trajectory import TrajectoryWriter as tw
from ase.io.trajectory import TrajectoryReader as tr
from amptorch.ase_utils import AMPtorch
from amptorch.trainer import AtomsTrainer

import numpy as np
import argparse
import sys

train_images = tr(
    "/users/chuang25/data/chuang25/github_repo/ml_qmc_force/examples/ch3cl/dmc/pyscf/test_dmc.traj"
)
# test_images = tr(
# "/users/chuang25/data/chuang25/github_repo/ml_qmc_force/examples/h2o/dmc_1000/dmc_1200blocks.traj"
# "/users/chuang25/data/chuang25/github_repo/ml_qmc_force/examples/h2o/dmc/dmc_sample1000.traj"
# )

print(len(train_images))

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
trainer.train()
# trainer.load_pretrained(
# "/users/chuang25/data/chuang25/github_repo/ml_qmc_force/examples/h2o/amptorch_dmc/200blocks/checkpoints/2022-02-07-14-31-05-test"
# )

predictions = trainer.predict(train_images)

true_energies = np.array([image.get_potential_energy() for image in train_images])
pred_energies = np.array(predictions["energy"])
pred_forces = np.array(predictions["forces"])

print("This is TRAIN: ", len(true_energies))
print("TRAIN Energy MSE:", np.mean((true_energies - pred_energies) ** 2) / 5)
print("TRAIN Energy MAE:", np.mean(np.abs(true_energies - pred_energies)) / 5)
print("TRAIN Energy RMSE:", np.sqrt(np.mean((true_energies - pred_energies) ** 2)) / 5)
