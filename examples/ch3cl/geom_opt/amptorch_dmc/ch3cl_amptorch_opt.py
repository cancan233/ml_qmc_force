import numpy as np
import torch
from ase import Atoms
from ase.io import read
from amptorch.ase_utils import AMPtorch
from amptorch.trainer import AtomsTrainer
from ase.optimize import QuasiNewton, BFGS

from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin

from ase.calculators.psi4 import Psi4

import numpy as np

from ase.io.trajectory import TrajectoryReader as tr


train_images = tr(
        "../../data/test_dmc.traj"
        )

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
"../../train/amptorch_dmc/2022-05-10-22-17-18-test")


calc = AMPtorch(trainer)

np.random.seed(1)

# images = tr(
#     "../dft/ch3cl_opt.traj"
# )[0]
images = tr(
    "./ch3cl_opt_bk.traj"
)[0]

atoms = images

print(atoms.get_positions())

atoms.set_calculator(calc)

# qn = QuasiNewton(atoms, trajectory="ch3cl_opt.traj")
qn = BFGS(atoms, trajectory="ch3cl_opt.traj", logfile="ch3cl_opt.log")
qn.run(fmax=0.05)