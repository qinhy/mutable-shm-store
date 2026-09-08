"""Usage: python examples/worker.py OBJECT_ID WRITE_TOKEN"""

import sys

import numpy as np

import mstore

object_id, token = sys.argv[1:3]
store = mstore.connect()
obj = store.open(object_id, token, mode="write")
arr = obj.numpy()

# In-place brightness bump; no whole-object IPC copy.
np.add(arr, 5, out=arr, casting="unsafe")
print(f"mutated {arr.nbytes / 1024 / 1024:.1f} MiB in shared memory")
