from ase.io.trajectory import TrajectoryReader as tr
from ase.io.trajectory import TrajectoryWriter as tw
import os

num_trajs = 2000
gap = 201

for i in range(0, num_trajs, gap):
    files = os.listdir("./calculate/trajs_{}_{}/".format(i, i + gap))
    for file in files:
        if file[-4:] == ".out":
            with open("./calculate/trajs_{}_{}/".format(i, i + gap) + file) as f:
                lines = f.readlines()
            for line in lines:
                if line == "SCF not converged.":
                    print("found in trajs_{}_{}".format(i, i + gap))
                    break
