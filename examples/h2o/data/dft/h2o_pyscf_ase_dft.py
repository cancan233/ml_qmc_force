from pyscf_calc import parameters, PYSCF

import pyscf
from pyscf import dft
from ase.io.trajectory import TrajectoryReader as tr
from ase.io.trajectory import TrajectoryWriter as tw

from ase.units import Ha, Bohr, Debye
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
import jsonpickle

# images = tr(
#     "/users/chuang25/data/chuang25/github_repo/ml_qmc_force/examples/ch3cl/dft/pyscf/dft_force.traj"
# )

images = tr("./sequence.traj")
trajs_filename = "sequence_with_forces.traj"
trajs = tw(trajs_filename)

for i in range(len(images)):
    # for i in range(0, 2):
    atoms = images[i]
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

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    trajs.write(atoms, energy=energy, forces=forces)

print(len(tr(trajs_filename)))

# print(atoms.get_forces())
# print(atoms.get_potential_energy())
# fmax = 1e-3 * (Ha / Bohr)

# dyn = LBFGS(atoms,logfile='opt.log',trajectory='opt.traj')
# dyn.run(fmax=fmax)
