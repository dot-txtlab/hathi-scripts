import subprocess
import os, sys, re
from numpy.random import choice

# to run this file, prepare a list of HTIDs to download in a text file, one per line
# loop over the file with Bash and run one instance of this program per HTID for parallel downloading
# or use gnu parallel to ease burden on Hathi's servers

def find_dir(path):
    for root in os.walk(path):
        if len(root[1]) == 0 and len(root[2]) > 5:
            return os.path.abspath(root[0])
        else:
            continue


htid_path = re.sub(r'\W+', "_", sys.argv[1])
dir = "/media/secure_volume/" + htid_path
subprocess.run("htrc download -o \"" + dir + "\" " + sys.argv[1], shell=True)
path = find_dir(dir)
if path:
    files = os.listdir(path)
    keep_files = choice(files, 10)
    for filename in files:
        if filename not in keep_files:
            os.remove(path + "/" + filename)
    print("Obtained and trimmed " + sys.argv[1])
else:
    print("Failure! Skipping.")
