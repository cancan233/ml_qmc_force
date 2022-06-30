from pyscf_calc import parameters, PYSCF
from joblib import Parallel, delayed

import pyscf
from ase import Atoms
from ase.optimize import LBFGS
from pyscf import scf
from pyscf import dft, grad
from ase.io.trajectory import TrajectoryReader as tr
from ase.optimize import QuasiNewton

from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase import units

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Ha, Bohr, Debye
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
import jsonpickle

np.random.seed(1)
images = tr(
"./h2o_opt.traj"
)[-1]

atoms = images
simtype = "nve"
timestep = 0.1
time = 20000
# time = 10
model = "dft"

mol = pyscf.M(
    atom=atoms_from_ase(atoms), ecp="ccecp", basis="ccecp-ccpvtz", spin=0, charge=0,
)

mf = dft.RKS(mol)
mf.density_fit()
mf.max_cycle = 200
mf.level_shift = 0.0
mf.conv_tol = 1e-10
mf.conv_check = True
mf.xc = "pbe"

p = parameters()
index = 0
p.mode = "dft"
p.verbose = 3
p.show()

mf.verbose = p.verbose
atoms.calc = PYSCF(mf=mf, p=p)

# qn = QuasiNewton(atoms, trajectory="h2o_opt.traj", logfile="h2o_opt.log")
# qn.run(fmax=0.05)

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

MaxwellBoltzmannDistribution(atoms, 300 * units.kB)
Stationary(atoms)
ZeroRotation(atoms)
dyn.run(time)

