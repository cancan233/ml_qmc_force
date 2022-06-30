from pyscf_calc import parameters, PYSCF
from joblib import Parallel, delayed

import pyscf
import ase
from ase import Atoms
from ase.optimize import BFGS
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

# atoms = ase.build.molecule("H2O")
# print(atoms.get_positions())

# atoms.rattle(seed=1, stdev=0.1)

# print(atoms.get_positions())
# print(atoms.get_distances(0, [1, 2]))
# print(atoms.get_angle(2, 0, 1))

atoms = tr("../amptorch_dmc/h2o_opt.traj")[0]

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

qn = BFGS(atoms, trajectory="h2o_opt.traj", logfile="h2o_opt.log")
qn.run(fmax=0.05)
