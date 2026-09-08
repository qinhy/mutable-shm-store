"""Run the server first: mstore-server"""

import numpy as np

import mstore

store = mstore.connect()

image = store.create(shape=(1080, 1920, 3), dtype=np.uint8, metadata={"kind": "image"})
image.numpy().fill(10)

read_token = image.issue("read")
write_token = image.issue("write")

print("object:", image.object_id)
print("read token:", read_token)
print("write token:", write_token)
print("first pixel:", image.numpy()[0, 0])
