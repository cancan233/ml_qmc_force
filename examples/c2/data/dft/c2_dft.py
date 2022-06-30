#! /usr/bin/env python3

from nexus import settings, job, run_project
from nexus import generate_physical_system
from nexus import generate_pwscf
from nexus import generate_pw2qmcpack
from nexus import generate_qmcpack
from nexus import loop, linear, vmc, dmc
import numpy as np
from ase.io import read


# nexus settings
settings(
    pseudo_dir="../../../pseudopotentials",
    runs="",
    results="",
    status_only=0,
    generate_only=0,
    sleep=3,
    machine="ws16",
)

# create opt & DMC sim's for each bond length
scales = [
    1.00,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.8,
    0.85,
    0.90,
    0.925,
    0.95,
    0.975,
    1.025,
    1.05,
    1.075,
    1.10,
    1.15,
    1.20,
    1.25,
    1.30,
    1.35,
    1.40,
    1.45,
    1.50,
    1.75,
    2.0,
    2.25,
    2.5,
    2.75,
    3.0,
    3.25,
    3.5,
    3.75,
    4.0,
]

sims = []

for scale in scales:
    directory = "./scale/scale_" + str(scale)

    # make stretched/compressed dimer
    dimer = generate_physical_system(
        type="dimer",  # dimer selected
        dimer=("C", "C"),  # atoms in dimer
        separation=1.242 * scale,  # dimer bond length
        Lbox=15.0,  # box size
        units="A",  # Angstrom units
        net_spin=0,  # Nup-Ndown = 2
        C=4,  # pseudo O has 6 val. electrons
    )

    # describe scf run
    scf = generate_pwscf(
        identifier="scf",
        path=directory,
        system=dimer,
        job=job(cores=16),
        input_type="scf",
        pseudos=["C.BFD.upf"],
        input_dft="lda",
        ecut=150,
        degauss=0.0002,
        conv_thr=1e-7,
        mixing_beta=0.7,
        nosym=True,
        wf_collect=True,
    )
    sims.append(scf)

run_project(sims)
