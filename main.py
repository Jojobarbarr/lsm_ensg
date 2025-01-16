from mesh import Mesh
import importlib
import scipy.sparse
from scipy.sparse.linalg import lsmr

model = "ifp2"
m = Mesh(model+"/slice.obj")
attr = importlib.import_module(model + ".attributes")

m.write_vtk("output_ref.vtk")
m.write_obj("output_ref.obj")

# Lists of list of vertices on the same horizon.
# Assuming that the horizon_id are consecutive starting from 0.
vertices_on_horizons = [
    set() for horizon_id in set(attr.horizon_id) if horizon_id >= 0
]

# List of the vertices on a fault
vertices_on_faults = set()

for corner in range(m.ncorners):
    org_vertex = m.org(corner)
    if attr.horizon_id[corner] >= 0:
        vertices_on_horizons[attr.horizon_id[corner]].add(org_vertex)
    if attr.is_fault[corner]:
        vertices_on_faults.add(org_vertex)
        # As faults can be a border, we need to add the destination vertex as
        # well because it could be only the destination vertex of a fault
        # corner. As we are working with set, dupllicata are not an issue.
        vertices_on_faults.add(m.dst(corner))

# List of intersection lists between vertices on horizons and vertices on
# faults.
vertices_on_faulted_horizons = []
for vertices_on_horizon in vertices_on_horizons:
    # Find intersection with fault vertices
    faulted_vertices = vertices_on_horizon & vertices_on_faults
    vertices_on_faulted_horizons.append(list(faulted_vertices))

# Number of additionnal constraints
n_additional_constraints = sum(
    max(len(vertices_on_faulted_horizon) - 1, 0)
    for vertices_on_faulted_horizon
    in vertices_on_faulted_horizons
)

# Flatten the horizons
dim = 1  # We work on y-coordinate
A = scipy.sparse.lil_matrix(
    (m.ncorners + n_additional_constraints, m.nverts)
)
b = [0] * A.shape[0]

weight_keeping_half_edge_length = 1
weight_flattening_continuous_horizons = 100
weight_flattening_faulted_horizons = 10

# Continuous horizon part
for c in range(m.ncorners):
    distance = m.V[m.dst(c)][dim] - m.V[m.org(c)][dim]

    A[c, m.dst(c)] = weight_keeping_half_edge_length
    A[c, m.org(c)] = -weight_keeping_half_edge_length

    b[c] = distance * weight_keeping_half_edge_length

    if attr.horizon_id[c] >= 0:
        A[c, m.org(c)] = weight_flattening_continuous_horizons
        A[c, m.dst(c)] = -weight_flattening_continuous_horizons
        b[c] = 0

# Faulted horizon part
offset = 0
for horizon_id, vertices_on_faulted_horizon in enumerate(
    vertices_on_faulted_horizons
):
    for vertex_idx in range(len(vertices_on_faulted_horizon) - 1):
        A[
            m.ncorners + offset + vertex_idx,
            vertices_on_faulted_horizon[vertex_idx]
        ] = weight_flattening_faulted_horizons
        A[
            m.ncorners + offset + vertex_idx,
            vertices_on_faulted_horizon[vertex_idx + 1]
        ] = -weight_flattening_faulted_horizons

        b[m.ncorners + offset + vertex_idx] = 0

    offset += len(vertices_on_faulted_horizon) - 1

A = A.tocsr()
x = lsmr(A, b)[0]
for i in range(m.nverts):
    m.V[i][dim] = x[i]

# Flatten the faults
dim = 0  # We work on x-coordinate
A = scipy.sparse.lil_matrix((2 * m.ncorners, m.nverts))
b = [0] * A.shape[0]

weight_keeping_half_edge_length
weight_flattening_faults = 100
weight_connecting_faults = 1
for c in range(m.ncorners):
    distance = m.V[m.dst(c)][0] - m.V[m.org(c)][0]

    A[c, m.dst(c)] = weight_keeping_half_edge_length
    A[c, m.org(c)] = -weight_keeping_half_edge_length

    b[c] = distance * weight_keeping_half_edge_length

    if attr.is_fault[c]:
        A[c, m.dst(c)] = weight_flattening_faults
        A[c, m.org(c)] = -weight_flattening_faults

        b[c] = 0

        if attr.fault_opposite[c] > -1:
            A[2 * c, m.dst(attr.fault_opposite[c])] = weight_connecting_faults
            A[2 * c, m.org(c)] = -weight_connecting_faults
            b[c] = 0

A = A.tocsr()
x = lsmr(A, b)[0]
for i in range(m.nverts):
    m.V[i][dim] = x[i]

m.write_obj("output.obj")
m.write_vtk("output.vtk")
