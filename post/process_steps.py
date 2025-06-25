from get_sim_info import *
from grid_to_vtk import *
from get_sim_info import __macr_names__
import math
import sys
import os

if len(sys.argv) < 3:
    print("Usage: python3 example_vtk.py <id> <velocity_set>")
    sys.exit(1)

simulation_id = sys.argv[1]
velocity_set = sys.argv[2]

path = f"./../bin/{velocity_set}/{simulation_id}/"

if not os.path.exists(path):
    print(f"Error: Path {path} does not exist.")
    sys.exit(1)

macr_steps = get_macr_steps(path)
info = retrieve_sim_info(path)

for step in macr_steps:
    macr = get_macrs_from_step(step, path)
    print("Processing step", step)
    save_vtk_3d(macr, path, info['ID'] + "macr" + str(step).zfill(6), points=True)

    # deletes binary files as they are succesfully processed  
    for macr_name in __macr_names__:
        filenames = get_filenames_macr(macr_name, path)
        step_filename = [f for f in filenames if f"{macr_name}{step:06d}.bin" in f]
        for file_to_delete in step_filename:
            os.remove(file_to_delete)
            print(f"Deleted {file_to_delete}")
