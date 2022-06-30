from ase.io.trajectory import TrajectoryReader as tr
from ase.io.trajectory import TrajectoryWriter as tw

trajs = tr(
    "/users/chuang25/data/chuang25/github_repo/MachineLearningQMC/examples/ch3cl/ch3cl_pes/VTZ_with_fake_forces.traj"
)

print(len(trajs))

gap = 200

combine_trajs = []
for i in range(0, len(trajs), gap):
    # for i in range(0, 1000, gap):
    trajfile = "/users/chuang25/scratch/ml_qmc_force/ch3cl_dft_pyscf_ase_force/trajs_{}_{}/sequence_with_forces.traj".format(
        i, i + gap
    )
    combine_trajs += tr(trajfile)[1:]


trajwriter = tw("./dft_pyscf_ase_force.traj")
for traj in combine_trajs:
    trajwriter.write(traj)

print(len(tr("./dft_pyscf_ase_force.traj")))
