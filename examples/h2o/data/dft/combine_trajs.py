from ase.io.trajectory import TrajectoryReader as tr
from ase.io.trajectory import TrajectoryWriter as tw

num_trajs = 2000
gap = 201

combine_trajs = []
for i in range(0, num_trajs, gap):
    # print(i, i+gap)
    trajfile = "./calculate/trajs_{}_{}/sequence_with_forces.traj".format(
        i, i + gap
    )
    if i != 0:
        combine_trajs += tr(trajfile)[1:]
    else:
        combine_trajs += tr(trajfile)


trajwriter = tw("./test_dft_pyscf_ase_force.traj")
for traj in combine_trajs:
    trajwriter.write(traj)

print(len(tr("./test_dft_pyscf_ase_force.traj")))
