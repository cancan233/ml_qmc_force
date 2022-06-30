from pyscf_calc import parameters, PYSCF
from joblib import Parallel, delayed

import pyscf
from ase import Atoms
from ase.optimize import LBFGS
from pyscf import scf
from pyscf import dft, grad
from ase.io.trajectory import TrajectoryReader as tr
from ase.optimize import QuasiNewton, BFGS

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Ha, Bohr, Debye
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
import jsonpickle

np.random.seed(1)
images = tr(
    "../amptorch_dmc/ch3cl_opt_bk.traj"
)[0]

atoms = images
print(atoms.get_positions())


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

# print(atoms.get_forces())
# print(atoms.get_potential_energy())
# fmax = 1e-3 * (Ha / Bohr)

# qn = QuasiNewton(atoms, trajectory="ch3cl_opt.traj", logfile="ch3cl_opt.log")
qn = BFGS(atoms, trajectory="ch3cl_opt.traj", logfile="ch3cl_opt.log")
# qn.run(fmax=0.05)
qn.run(fmax=0.05)