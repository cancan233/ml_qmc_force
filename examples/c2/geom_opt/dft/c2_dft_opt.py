import numpy as np
import torch
from ase import Atoms
from ase.io import read
from ase.calculators.espresso import Espresso

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

atoms = images
model = "dft"

print(atoms.get_distances(0, 1))
atoms.set_distance(0, 1, 2)
print(atoms.get_distances(0, 1))
pseudopotentials = {
    "C": "C.BFD.upf",
}

dft_calc = Espresso(
    pseudopotentials=pseudopotentials,
    tstress=True,
    tprnfor=True,
    calculation="scf",
    disk_io="low",
    degauss=0.0002,
    ecutrho=600,
    kpts=(1, 1, 1),
    koffset=(0, 0, 0),
    ecutwfc=150,
    input_dft="lda",
    occupations="smearing",
    smearing="fermi-dirac",
    tot_charge=0,
    mixing_beta=0.7,
    mixing_mode="plain",
    conv_thr=1e-07,
    diagonalization="david",
    electron_maxstep=1000,
    nosym=True,
    verbosity="high",
)

atoms.set_calculator(dft_calc)
dyn = QuasiNewton(atoms, trajectory="c2_opt.traj")
dyn.run(fmax=0.05)

print(atoms.get_distances(0, 1))
