from pyscf import scf
from pyscf import dft, grad
# Nexus expands this with Mole info
$system

mf = dft.RKS(mol)
mf.density_fit()
mf.max_cycle = 200
mf.level_shift=0.0
mf.conv_tol=1e-10
mf.conv_check=True
mf.xc = "pbe"
mf.run() 
grad.RKS(mf).kernel()

