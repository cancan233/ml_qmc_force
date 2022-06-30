from ase.io import read, write
import numpy as np
import os
import subprocess
from ase.io.trajectory import TrajectoryReader as tr
from ase.io.trajectory import TrajectoryWriter as tw
import copy
import sys

from joblib import Parallel, delayed


def combine_trajs(trajectory_files):
    trajs = []
    for file in trajectory_files:
        traj = read(file, index=":")
        trajs += traj
    return trajs


def qmca_output(data_folder, num):
    file_path = "{}/dmc/{}_dmc.s001.scalar.dat".format(data_folder, num)
    if os.path.exists(file_path):
        output = subprocess.check_output(
            "/gpfs/data/brubenst/chuang25/local/qmcpack/nexus/bin/qmca -e 50 -q e -u eV {}".format(
                file_path
            ),
            shell=True,
        )
        dmc_energy, uncertainty = float(output.split()[5]), float(output.split()[7])
    else:
        dmc_energy, uncertainty = 0, 0

    # file_path = "{}/dft/{}_scf.out".format(data_folder, num)
    # if os.path.exists(file_path):
    #     with open(file_path, "r") as file:
    #         lines = file.readlines()
    #     dft_forces = []
    #     for i in range(len(lines)):
    #         if "!" == lines[i][0]:
    #             dft_energy = float(lines[i].split()[4]) * 13.605662285137
    #         if "Forces acting on atoms (cartesian axes, Ry/au)" in lines[i]:
    #             dft_forces.append(
    #                 [float(i) * 25.71104309541616 for i in lines[i + 2].split()[-3:]]
    #             )
    #             dft_forces.append(
    #                 [float(i) * 25.71104309541616 for i in lines[i + 3].split()[-3:]]
    #             )
    #             dft_forces.append(
    #                 [float(i) * 25.71104309541616 for i in lines[i + 4].split()[-3:]]
    #             )
    # else:
    #     dft_energy, dft_forces = 0, 0

    # return dmc_energy, uncertainty, dft_energy, dft_forces, num

    return dmc_energy, uncertainty, num

def main():

    # define variables
    set_num = 3
    trajectory_folder = "../trajs_set{}".format(set_num)
    data_folder = "./results"
    total_trajectory_num = 60
    num_blocks = 400
    dmc_trajs = tw("dmc_{}blocks_set{}.traj".format(num_blocks, set_num))
    dft_trajs = tw("dft_{}blocks_set{}.traj".format(num_blocks, set_num))

    trajectory_files = []
    for i in range(total_trajectory_num):
        trajectory_files.append(
            "./{}/trajs_{}_{}.traj".format(trajectory_folder, i * 201, i * 201 + 201)
        )
    trajs = combine_trajs(trajectory_files)

    print(len(trajs))

    dmc_files = sorted(os.listdir("./{}/dmc".format(data_folder, num_blocks)))
    output = Parallel(n_jobs=-1)(
        delayed(qmca_output)(data_folder, int(file.split("_")[0])) for file in dmc_files
    )

    print("extracted {} points".format(len(output)))

    fake_force = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]

    for i in range(len(output)):
        # dmc_energy, uncertainty, dft_energy, dft_forces, num = (
        #     output[i][0],
        #     output[i][1],
        #     output[i][2],
        #     output[i][3],
        #     output[i][4],
        # )

#         if dmc_energy != 0 and uncertainty != 0 and dft_energy != 0 and dft_forces != 0:
#             temp_traj = copy.deepcopy(trajs[num])
#             temp_traj.info["uncertainty"] = uncertainty
#             dmc_trajs.write(temp_traj, energy=dmc_energy, forces=fake_force)

#             temp_traj = copy.deepcopy(trajs[num])
#             dft_trajs.write(
#                 temp_traj, energy=dft_energy, forces=dft_forces,
#             )

        dmc_energy, uncertainty, num = output[i][0], output[i][1], output[i][2]
        if dmc_energy != 0 and uncertainty != 0:
            temp_traj = copy.deepcopy(trajs[num])
            temp_traj.info["uncertainty"] = uncertainty
            dmc_trajs.write(temp_traj, energy=dmc_energy, forces=fake_force)


if __name__ == "__main__":
    main()
