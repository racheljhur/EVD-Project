import numpy as np

pc_scores = np.load('pc_scores.npy', mmap_mode = 'r')
exc_vals = np.load('responses.npy', mmap_mode = 'r')
print(pc_scores.shape), print(exc_vals.shape)

exc_vals = exc_vals.reshape(15 * 200 * 56, 1)
pc_scores = np.repeat(pc_scores, 56, axis=1)

pc_scores = pc_scores.reshape(15 * 200 * 56, 10)

exc_vals = exc_vals.reshape(15, 200 * 56, 1)
pc_scores = pc_scores.reshape(15, 200 * 56, 10)
print(pc_scores.shape), print(exc_vals.shape)

np.save('tiled_pc_scores.npy', pc_scores), np.save('flattened_exc_vals.npy', exc_vals)
