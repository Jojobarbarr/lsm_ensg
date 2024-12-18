from mesh import Mesh
import importlib
import numpy as np
import scipy.sparse
from scipy.sparse.linalg import lsmr

model = "chevron"
m = Mesh(model+"/slice.obj")
attr = importlib.import_module(model + ".attributes")

m.write_vtk("output_ref.vtk")
m.write_obj("output_ref.obj")

horizon_set = {attr.horizon_id[c] for c in range(m.ncorners) if attr.horizon_id[c] >= 0}
sorted_horizon_list = sorted(list(horizon_set))
horizon_list = [[] for horizon_id in sorted_horizon_list]

dim = 1
A = scipy.sparse.lil_matrix((m.ncorners, m.nverts))
b = [0] * A.shape[0]

row = 0
weight = 100
for c in range(m.ncorners):
    distance = m.V[m.dst(c)][dim] - m.V[m.org(c)][dim]
    A[c, m.dst(c)] = 1
    A[c, m.org(c)] = -1
    b[c] = distance
    if attr.horizon_id[c] >= 0:
        A[c, m.org(c)] = weight
        b[c] = (attr.horizon_id[c] / 37.76) * weight


print(f"A.shape = {A.shape}")
print(f"b.shape = {len(b)}")
A = A.tocsr() # convert to compressed sparse row format for faster matrix-vector muliplications
x = lsmr(A, b)[0] # call the least squares solver
for i in range(m.nverts): # apply the computed flattening
    m.V[i][dim] = x[i]

dim = 0
A = scipy.sparse.lil_matrix((2*m.ncorners, m.nverts))
b = [0] * A.shape[0]

row = 0

for c in range(m.ncorners):
    distance = m.V[m.dst(c)][0] - m.V[m.org(c)][0]
    A[c, m.dst(c)] = 1
    A[c, m.org(c)] = -1
    b[c] = distance * 1
    if attr.is_fault[c]:
        A[c, m.dst(c)] = 100
        A[c, m.org(c)] = -100
        b[c] = 0
        if attr.fault_opposite[c] > -1:
            A[2*c, m.dst(attr.fault_opposite[c])] = 1
            A[2*c, m.org(c)] = -1
            b[c] = 0


print(f"A.shape = {A.shape}")
print(f"b.shape = {len(b)}")
A = A.tocsr() # convert to compressed sparse row format for faster matrix-vector muliplications
x = lsmr(A, b)[0] # call the least squares solver
for i in range(m.nverts): # apply the computed flattening
    m.V[i][dim] = x[i]


m.write_obj("output.obj")
m.write_vtk("output.vtk")

