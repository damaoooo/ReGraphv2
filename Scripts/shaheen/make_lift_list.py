"""Write the list of input binaries under a dataset dir (same filter as task1_lift.py).

    .venv/bin/python Scripts/shaheen/make_lift_list.py Binaries/Dataset-1 > lift_inputs.txt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task1_lift import is_input_binary_file  # noqa: E402

root = os.path.abspath(sys.argv[1])
for dirpath, _, files in os.walk(root):
    for name in sorted(files):
        if is_input_binary_file(name):
            print(os.path.join(dirpath, name))
