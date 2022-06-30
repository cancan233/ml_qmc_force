#! /usr/bin/env python3

from nexus import settings, job, run_project, obj
from nexus import ppset
from nexus import generate_physical_system
from nexus import generate_pyscf
from nexus import generate_convert4qmc
from nexus import generate_qmcpack
from nexus import loop, linear, vmc, dmc

from ase.io.trajectory import TrajectoryReader as tr
from ase.io.trajectory import TrajectoryWriter as tw
from ase.io import read, write
import copy
import subprocess
import sys
import numpy as np

# Obtain the core count of the local machine (lab only)
import os


def main():
    settings(
        pseudo_dir="/users/chuang25/scratch/ml_qmc_force/h2o_data_revisit_pyscf/00_data/dmc/pseudopotentials",
        results="",
        sleep=3,
        machine="ws22",
    )

    ppset(
        label="ccecp", qmcpack=["H.ccECP.xml", "O.ccECP.xml"],
    )

    trajs = tr("./sequence.traj")

    os.mkdir("./structures/")

    sims = []
    for num in range(len(trajs)):
    # for num in range(1):

        # sims = []
        pos = trajs[num].get_positions()
        structure_file = "./structures/h2o_{}.xyz".format(num)
        directory = "h2o_{}/".format(num)

        with open(structure_file, "w") as f:
            f.write("3\n\n")
            f.write("O\t{}\t{}\t{}\t\n".format(pos[0][0], pos[0][1], pos[0][2]))
            f.write("H\t{}\t{}\t{}\t\n".format(pos[1][0], pos[1][1], pos[1][2]))
            f.write("H\t{}\t{}\t{}\t\n".format(pos[2][0], pos[2][1], pos[2][2]))

        system = generate_physical_system(structure=structure_file, H=1, O=6,)

        # Perform Hartree-Fock
        scf = generate_pyscf(
            identifier="scf",  # Log output goes to scf.out
            path=directory + "scf",  # Directory to run in
            job=job(serial=True, app="python3"),
            template="./scf_template.py",  # PySCF template file
            system=system,
            mole=obj(  # Used to make Mole() inputs
                ecp="ccecp", basis="ccecp-ccpvtz", symmetry=True, verbose=5
            ),
            save_qmc=True,  # Save wfn data for qmcpack
        )
        sims.append(scf)

        # convert orbitals to QMCPACK format
        c4q = generate_convert4qmc(
            identifier="c4q",
            path=directory + "scf",  # directory to run in
            job=job(cores=1),
            dependencies=(scf, "orbitals"),  # Create a dependency to DFT success
        )
        sims.append(c4q)

        # collect dependencies relating to orbitals
        orbdeps = [
            (c4q, "particles"),  # pyscf changes particle positions
            (c4q, "orbitals"),
        ]

        if num == 0:
            # optimize 2-body Jastrow
            optJ2 = generate_qmcpack(
                identifier="opt",
                path=directory + "optJ2",  # directory to run in
                job=job(cores=16),
                use_nonlocalpp_deriv="no",
                system=system,
                pseudos="ccecp",
                J2=True,  # 2-body B-spline Jastrow
                J1_rcut=6.0,  # 6 Bohr cutoff for J1
                J2_rcut=8.0,  # 8 Bohr cutoff for J2
                # seed=42,  # Fix the seed (lab only)
                qmc="opt",  # Wavefunction optimization run
                minmethod="oneshift",  # Energy minimization
                init_cycles=4,  # 4 iterations allowing larger parameter changes
                init_samples=25600,  # VMC samples per iteration
                init_minwalkers=0.1,
                cycles=10,  # 10 production iterations
                blocks=40,
                warmupsteps=5,
                timestep=0.5,
                samples=25600,  # VMC samples per iteration
                substeps=5,
                minwalkers=0.5,
                dependencies=orbdeps,
            )
            # optJ2 = generate_qmcpack(
            #     identifier="optJ2",
            #     path=directory + "optJ2",
            #     job=job(cores=16, app="qmcpack"),
            #     pseudos=["H.ccECP.xml", "C.ccECP.xml", "Cl.ccECP.xml"],
            #     system=system,
            #     input_type="basic",
            #     corrections=[],
            #     jastrows=[("J1", "bspline", 8, 6), ("J2", "bspline", 10, 8)],
            #     calculations=[
            #         loop(
            #             max=6,
            #             qmc=linear(
            #                 energy=0.0,
            #                 unreweightedvariance=1.0,
            #                 reweightedvariance=0.0,
            #                 timestep=0.5,
            #                 warmupsteps=100,
            #                 samples=25600,
            #                 stepsbetweensamples=10,
            #                 blocks=10,
            #                 minwalkers=0.1,
            #                 bigchange=15.0,
            #                 alloweddifference=1e-4,
            #             ),
            #         )
            #     ],
            #     dependencies=orbdeps,
            # )

            sims.append(optJ2)

            # optimize 3-body Jastrow
            optJ3 = generate_qmcpack(
                identifier="opt",
                path=directory + "optJ3",  # directory to run in
                job=job(cores=16),
                system=system,
                pseudos="ccecp",
                J3=True,  # 3-body B-spline Jastrow
                J3_rcut=3,  # 3-body B-spline Jastrow
                # seed=42,  # Fix the seed (lab only)
                qmc="opt",  # Wavefunction optimization run
                minmethod="oneshift",  # Energy minimization
                init_cycles=4,  # 4 iterations allowing larger parameter changes
                init_samples=25600,  # VMC samples per iteration
                init_minwalkers=0.1,
                cycles=10,  # 10 production iterations
                blocks=40,
                warmupsteps=5,
                timestep=0.5,
                samples=160000,  # VMC samples per iteration
                steps=5,
                substeps=5,
                minwalkers=0.5,
                dependencies=orbdeps
                + [(optJ2, "jastrow")],  # Dependece (1B and 2B Jastrows)
            )
            # optimize 3-body Jastrow
            # optJ3 = generate_qmcpack(
            #     identifier="optJ3",
            #     path=directory + "optJ3",
            #     job=job(cores=16, app="qmcpack"),
            #     pseudos=["H.ccECP.xml", "C.ccECP.xml", "Cl.ccECP.xml"],
            #     system=system,
            #     J3=True,
            #     J3_rcut=3,
            #     calculations=[
            #         loop(
            #             max=6,
            #             qmc=linear(
            #                 energy=0.0,
            #                 unreweightedvariance=1.0,
            #                 reweightedvariance=0.0,
            #                 timestep=0.5,
            #                 warmupsteps=100,
            #                 samples=25600,
            #                 stepsbetweensamples=10,
            #                 blocks=40,
            #                 minwalkers=0.1,
            #                 bigchange=15.0,
            #                 alloweddifference=1e-4,
            #             ),
            #         )
            #     ],
            #     dependencies=orbdeps + [(optJ2, "jastrow")],
            # )

            sims.append(optJ3)

        # run DMC with 1,2 and 3 Body Jastrow function
        qmc = generate_qmcpack(
            identifier="dmc",
            # seed=42,
            path=directory + "dmc",  # directory to run in
            job=job(cores=16),  # Submit with the number of cores available
            system=system,
            pseudos="ccecp",
            jastrows=[],
            qmc="dmc",  # dmc run
            vmc_samples=2048,  # Number of Samples (selected from a VMC step)
            vmc_warmupsteps=100,  # Number of Equilibration steps
            warmupsteps=800,  # Number of Equilibration steps
            vmc_blocks=200,  # Number of VMC blocks (To generate the DMC samples)
            vmc_steps=20,  # Number of VMC steps (To generate DMC samples)
            vmc_timestep=0.1,  # VMC Timestep (To Generate DMC samples)
            timestep=0.01,  # DMC timestep
            steps=40,  # start with small number for large timesteps [autocorrelation]
            blocks=400,  # Number of DMC blocks
            nonlocalmoves="v3",
            dependencies=orbdeps
            + [(optJ3, "jastrow")],  # Dependece (1B 2B and 3B  Jastrows)
        )

        sims.append(qmc)

        # run_project(sims)
    run_project(sims)


if __name__ == "__main__":
    main()
