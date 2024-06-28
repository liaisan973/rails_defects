import numpy as np
import time

start_time = time.time()
matrix1 = np.random.rand(1280, 960)
matrix2 = np.random.rand(960, 1000)
result_matrix_multiplication = np.dot(matrix1, matrix2)
end_time = time.time()
processing_time = end_time - start_time
print("Processing Time: {:.4f} seconds".format(processing_time))

