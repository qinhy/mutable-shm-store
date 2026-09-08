"""Usage: python examples/reader.py OBJECT_ID READ_TOKEN"""

import sys

import mstore

object_id, token = sys.argv[1:3]
obj = mstore.connect().open(object_id, token, mode="read")
arr = obj.numpy()
print("shape:", arr.shape, "dtype:", arr.dtype, "writeable:", arr.flags.writeable)
print("first pixel:", arr[0, 0])
