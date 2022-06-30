from ase.io import read, write
import numpy as np
import os
import subprocess
from ase.io.trajectory import TrajectoryReader as tr
from ase.io.trajectory import TrajectoryWriter as tw
import copy
import sys

from joblib import Parallel, delayed


def qmca_output(num):
    if num % 1000 == 0:
        print("reaching point {}".format(num))

    file_path = "./data/dmc/{}_dmc.s001.scalar.dat".format(num)
    if os.path.exists(file_path):
        output = subprocess.check_output(
            "/gpfs/data/brubenst/chuang25/local/qmcpack/nexus/bin/qmca -q e -u eV {}".format(
                file_path
            ),
            shell=True,
        )
        dmc_energy, uncertainty = float(output.split()[5]), float(output.split()[7])
    else:
        dmc_energy, uncertainty = 0, 0
    return dmc_energy, uncertainty, num


def main():
    data_trajs = tr(
        "/users/chuang25/data/chuang25/github_repo/MachineLearningQMC/examples/ch3cl/ch3cl_pes/VTZ_with_fake_forces.traj"
    )
    dmc_trajs_filename = "dmc.traj"
    dmc_trajs = tw(dmc_trajs_filename)

    output = Parallel(n_jobs=-1)(
        delayed(qmca_output)(num) for num in range(len(data_trajs))
    )

    print("extracted {} points".format(len(output)))

    fake_forces = [
        [0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1],
    ]

    for i in range(len(output)):
        dmc_energy, uncertainty, num = output[i][0], output[i][1], output[i][2]
        if dmc_energy != 0 and uncertainty != 0:
            temp_traj = copy.deepcopy(data_trajs[num])
            temp_traj.info["uncertainty"] = uncertainty
            dmc_trajs.write(temp_traj, energy=dmc_energy, forces=fake_forces)

    print(len(tr(dmc_trajs_filename)))


if __name__ == "__main__":
    main()
