import numpy as np
import torch
from ase import Atoms
from ase.io import read
from ase.calculators.espresso import Espresso
from ase.calculators.socketio import SocketIOCalculator

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
from ase.md import MDLogger
from ase.calculators.psi4 import Psi4

import numpy as np
import argparse
import sys

from ase.io.trajectory import TrajectoryReader as tr
from ase.io.trajectory import TrajectoryWriter as tw

np.random.seed(1)

images = tr(
    "/users/chuang25/data/chuang25/github_repo/ml_qmc_force/examples/c2/dft/widerange/dft.traj"
)[14]

atoms = images
simtype = "nve"
timestep = 0.1
time = 5000
model = "dft"

print(atoms.get_distances(0, 1))

pseudopotentials = {
    "C": "C.BFD.upf",
}

unixsocket = "ase_espresso"
# srun --mpi=pmix pw.x -in espresso.pwi > espresso.pwo
command = "srun --mpi=pmix pw.x --ipi {unixsocket}:UNIX -in espresso.pwi > espresso.pwo".format(
    unixsocket=unixsocket
)

espresso = Espresso(
    command=command,
    pseudopotentials=pseudopotentials,
    # pseudo_dir=pseudo_dir,
    tstress=True,
    tprnfor=True,
    calculation="scf",
    disk_io="low",
    degauss=0.0002,
    ecutrho=800,
    kpts=(1, 1, 1),
    koffset=(0, 0, 0),
    ecutwfc=200,
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

# atoms.set_calculator(espresso)
qn = QuasiNewton(atoms, trajectory="c2_opt.traj")

with SocketIOCalculator(espresso, log=sys.stdout, unixsocket=unixsocket) as calc:
    atoms.calc = calc
    qn.run(fmax=0.05)

print(atoms.get_distances(0, 1))

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

with SocketIOCalculator(espresso, log=sys.stdout, unixsocket=unixsocket) as calc:
    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, 300 * units.kB)
    Stationary(atoms)
    ZeroRotation(atoms)
    dyn.run(time)
