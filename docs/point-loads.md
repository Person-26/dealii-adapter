# Local extension: propeller and control-surface point loads

This fork of the deal.II-preCICE adapter extends the linear-elasticity solver
(`source/linear_elasticity`) with optional **point loads** received from the
fluid participant on small (single- or multi-vertex) meshes:

- **Propeller loads** — actuator-disk `Thrust` + `PropTorque` applied at the
  hub vertex.
- **Control-surface hinge loads** — `HingeForce` + `HingeMoment` applied at
  each surface hinge vertex.

These are in addition to the standard interface `Stress` (surface traction)
read and `Displacement` write. They let the structure feel the propeller and
elevon reaction forces without adding them to the distributed surface stress.

## Adapter additions (`include/adapter/adapter.h`)

- `configure_propeller(mesh, forceData, torqueData)` /
  `set_prop_mesh_access_region()` / `initialize_prop_mesh()` /
  `read_prop_data(t, force, torque)` — receive the `Propeller-Mesh`
  (api-access), read the per-vertex `Thrust`/`PropTorque` and expose the hub
  coordinates.
- `configure_hinges(mesh, forceData, momentData)` /
  `set_hinge_mesh_access_region()` / `initialize_hinge_mesh()` /
  `read_hinge_data(t, force, moment)` — same for the control-surface hinge
  mesh (`HingeForce`/`HingeMoment`).

## Solver additions (`source/linear_elasticity`)

- `add_point_force(rhs, point, force)` — distributes a point force to the DoFs
  of the cell containing `point` (via the shape functions at the point).
- `add_propeller_rhs()` — for each hub: point force (thrust) + the torque as a
  three-axis couple.
- `add_hinge_rhs()` — same for each hinge (force + moment).
- `add_propeller_rhs` / `add_hinge_rhs` are called from `assemble_rhs()`
  alongside the body force; the moment couples use a clamped arm (`h ≤ 4 mm`)
  so the two couple points stay inside the (possibly thin) structure.

## Parameters (`parameters.prm`)

Under `precice configuration`:

```text
set Enable propeller loads   = true
set Propeller mesh name      = Propeller-Mesh
set Propeller force data name  = Thrust
set Propeller torque data name = PropTorque

set Enable hinge loads       = true
set Hinge mesh name          = Control-Mesh
set Hinge force data name    = HingeForce
set Hinge moment data name   = HingeMoment
```

The hub/hinge vertices must lie **inside the solid mesh** for `add_point_force`
to find a containing cell (the `PF` scenario of the linear-elasticity solver
provides a large enough plate; the default tiny `FSI3` flap is too small).

See `../multiphysics/test/flying-sled/README.md` for a complete example.