from mesh import Mesh
import importlib

model = "ifp2"
# m = Mesh(model+"/slice.obj")
m = Mesh("output.obj")
attr = importlib.import_module(model + ".attributes")

for c in range(m.ncorners): # up hoziron vertices
    if attr.horizon_id[c] >= 0:
        height = (1 + attr.horizon_id[c]) / 37.76 # arbitrary scaling coefficent
        m.V[m.org(c)][2] = m.V[m.dst(c)][2] = height


for c in range(m.ncorners): # lower vertices in faults
    if attr.is_fault[c]:
        m.V[m.org(c)][2] -= 0.01431 # arbitrary scaling coefficent
        m.V[m.dst(c)][2] -= 0.01431 # to make the result look nice

m.write_vtk("output_attr.vtk")


m = Mesh("output_ref.obj")
attr = importlib.import_module(model + ".attributes")

for c in range(m.ncorners): # up hoziron vertices
    if attr.horizon_id[c] >= 0:
        height = (1 + attr.horizon_id[c]) / 37.76 # arbitrary scaling coefficent
        m.V[m.org(c)][2] = m.V[m.dst(c)][2] = height


for c in range(m.ncorners): # lower vertices in faults
    if attr.is_fault[c]:
        m.V[m.org(c)][2] -= 0.00431 # arbitrary scaling coefficent
        m.V[m.dst(c)][2] -= 0.00431 # to make the result look nice

m.write_vtk("output_ref_attr.vtk")
