#ifndef LINEAR_ELASTICITY_H
#define LINEAR_ELASTICITY_H

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/revision.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <adapter/adapter.h>
#include <adapter/parameters.h>
#include <adapter/time_handler.h>

#include <fstream>
#include <iostream>

#include "postprocessor.h"

// The Linear_Elasticity case includes a linear elastic material with a one-step
// theta time integration
namespace Linear_Elasticity
{
  using namespace dealii;

  template <int dim>
  class ElastoDynamics
  {
  public:
    ElastoDynamics(const std::string &parameter_file);

    ~ElastoDynamics();
    // As usual in dealii, the run function covers the main time loop of the
    // system
    void
    run();

  private:
    // Create the mesh and set boundary IDs for different boundary conditions
    void
    make_grid();

    // Build the 3D flying-sled plate (split at the hinge when servo-driven)
    void
    make_sled_grid();

    // Set up the FE system and allocate data structures
    void
    setup_system();

    // Compute time invariant matrices e.g. stiffness matrix and mass matrix
    void
    assemble_system();

    // Assemble the Neumann contribution i.e. the coupling data obtained from
    // the Fluid participant
    void
    assemble_rhs();

    void
    assemble_consistent_loading();

    // Add the propeller load (point force + couple) to the RHS
    void
    add_propeller_rhs();

    // Add the control-surface hinge load (point force + couple) to the RHS
    void
    add_hinge_rhs();

    // Build the actuated control-surface hinge: one rotational DOF per surface,
    // the flap hinge nodes tied to the airframe, a mass-orthogonal gauge that
    // removes the flap rigid rotation, and the servo's inertia/spring/damping
    // added to the stepping system. Requires the mesh and the elastic assembly
    // to be in place, so it is called after assemble_system().
    void
    setup_servo_hinges();

    // Add the servo inertia/spring/damping to the augmented stepping system.
    void
    add_servo_terms();

    // Write / read the state a restart needs (displacement, velocity, the
    // previous load and the servo states), see Parameters::Time.
    void
    write_checkpoint() const;
    void
    read_checkpoint(const std::string &file);

    // Advance the servo states by the time step and write theta back into the
    // elastic displacement, so the flap moves with the actuated hinge.
    void
    update_servo_hinges();

    // Add a point force (vector) applied at the given point to the RHS
    void
    add_point_force(Vector<double> &rhs,
                    const Point<dim> &point,
                    const Vector<double> &force);

    // Solve the linear system
    void
    solve();

    // Update the displacement according to the theta scheme
    void
    update_displacement();

    // Output results to vtk files
    void
    output_results() const;

    // Paramter class parsing all user specific input parameters
    const Parameters::AllParameters parameters;

    // Boundary IDs, reserved for the respectve application
    unsigned int       clamped_mesh_id;
    unsigned int       out_of_plane_clamped_mesh_id;
    const unsigned int interface_boundary_id;

    // Dealii typical objects
    Triangulation<dim>   triangulation;
    DoFHandler<dim>      dof_handler;
    FESystem<dim>        fe;
    MappingQGeneric<dim> mapping;
    const unsigned int   quad_order;

    AffineConstraints<double> hanging_node_constraints;

    // Hanging-node constraints merged with the servo hinge ties and the
    // mass-orthogonal gauge that removes the flap rigid rotation. Only used
    // when `servo_enabled`.
    AffineConstraints<double> servo_constraints;

    // Matrices used during computations
    SparsityPattern      sparsity_pattern;
    // Augmented (n_elastic_dofs + n_flaps) sparsity pattern: the elastic
    // pattern plus the fill-in of the servo gauge and the theta rows.
    SparsityPattern      augmented_sparsity_pattern;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> stiffness_matrix;
    SparseMatrix<double> system_matrix;
    SparseMatrix<double> stepping_matrix;

    // Time dependent variables
    Vector<double> old_velocity;
    Vector<double> velocity;
    Vector<double> old_displacement;
    Vector<double> displacement;
    Vector<double> old_stress;
    Vector<double> stress;
    Vector<double> system_rhs;

    // Body forces e.g. gravity. Values are specified in the input file
    const bool     body_force_enabled;
    Vector<double> body_force_vector;

    // Propeller loads received from the fluid participant
    bool             prop_enabled;
    std::vector<double> prop_force_values;
    std::vector<double> prop_torque_values;

    // Control-surface hinge loads received from the fluid participant
    bool                hinge_enabled;
    std::vector<double> hinge_force_values;
    std::vector<double> hinge_moment_values;

    // Actuated control-surface hinges (rotational DOF + servo). Each control
    // surface is its own flap: a separate mesh piece (see make_grid()) whose
    // hinge-line nodes are tied to the airframe. The flap keeps its elastic
    // bending, and its rigid rotation about the hinge is carried by `theta`, an
    // extra scalar DOF driven by the servo. A mass-orthogonal gauge removes the
    // flap's own rigid rotation from the elastic DOFs, so the system stays
    // symmetric positive definite.
    struct ServoHinge
    {
      Point<dim>            hinge_point; // point on the hinge line
      Point<dim>            axis;        // hinge line direction (unit)
      std::size_t           dof = 0;     // index of `theta` in the augmented system
      // The flap block, used to write the rigid rotation back to the mesh.
      std::vector<types::global_dof_index> flap_dofs;
      std::vector<Point<dim>>              flap_points;
      std::vector<unsigned int>            flap_components;
      // Coupling vector cg[j] = int rho phi_j (axis x (x - hinge)) dV.
      std::map<types::global_dof_index, double> cg;
      double                                    inertia = 0.0; // int rho g.g dV
    };

    // Augmented dof count: elastic dofs followed by one theta per flap (one
    // flap per control surface).
    unsigned int    n_elastic_dofs = 0;
    unsigned int    n_flaps        = 0;
    bool            servo_enabled;
    std::vector<ServoHinge> servo_hinges;
    // Per control surface, as read from the fluid.
    std::vector<double>     servo_command_values;
    std::vector<double>     servo_hinge_moment_axis;

    // The theta row's stepping-matrix entry (flap + reflected servo inertia).
    std::vector<double> servo_inertia_row;

    // Servo states kept as deal.II vectors so they can take part in the
    // implicit-coupling checkpoint/reload.
    Vector<double> servo_theta, servo_omega, servo_cmd;

    // In order to measure some timings
    mutable TimerOutput timer;

    // The main adapter objects: The time class keeps track of the current time
    // and time steps. The Adapter class includes all functionalities for
    // coupling via preCICE. Look at the documentation of the class for more
    // information.
    Adapter::Time                                                    time;
    Adapter::Adapter<dim, Vector<double>, Parameters::AllParameters> adapter;

    // Alias for all time dependent variables, which should be saved/reloaded
    // in case of an implicit coupling. This vector is directly used in the
    // Adapter class
    std::vector<Vector<double> *> state_variables;
  };
} // namespace Linear_Elasticity

#endif // LINEAR_ELASTICITY_H
