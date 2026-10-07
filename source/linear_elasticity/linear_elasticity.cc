#include "include/linear_elasticity.h"

#include <fstream>
#include <iomanip>
#include <sstream>

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

// rb::servoStep, the servo model shared with the multiphysics participants
// (actuators.hpp, found through MP_SHARED_INCLUDE_DIR).
#include <actuators.hpp>

#include <fstream>
#include <iostream>
#include <algorithm>
#include <limits>
#include <locale>
#include <map>
#include <numeric>
#include <set>
#include <sstream>

#include "include/postprocessor.h"

// The Linear_Elasticity case includes a linear elastic material with a one-step
// theta time integration
namespace Linear_Elasticity
{
  using namespace dealii;

  // Constructor
  template <int dim>
  ElastoDynamics<dim>::ElastoDynamics(const std::string &parameter_file)
    : parameters(parameter_file)
    , interface_boundary_id(6)
    , dof_handler(triangulation)
    , fe(FE_Q<dim>(parameters.poly_degree), dim)
    , mapping(MappingQGeneric<dim>(parameters.poly_degree))
    , quad_order(parameters.poly_degree + 1)
    , body_force_enabled(parameters.body_force.norm() > 1e-15)
    , prop_enabled(parameters.prop_enabled)
    , hinge_enabled(parameters.hinge_enabled)
    , servo_enabled(parameters.servo_enabled)
    , timer(std::cout, TimerOutput::summary, TimerOutput::wall_times)
    , time(parameters.end_time, parameters.delta_t)
    , adapter(parameters, interface_boundary_id)
  {}



  // Destructor
  template <int dim>
  ElastoDynamics<dim>::~ElastoDynamics()
  {
    dof_handler.clear();
  }



  template <int dim>
  void
  ElastoDynamics<dim>::make_grid()
  {
    uint n_x, n_y, n_z;

    // All preconfigured cases consist of a rectangular block
    Point<dim> point_bottom;
    Point<dim> point_tip;

    // boundary IDs are obtained through colorize = true
    uint id_flap_long_bottom, id_flap_long_top, id_flap_short_bottom,
      id_flap_short_top, id_flap_out_of_plane_bottom, id_flap_out_of_plane_top;

    // Hron & Turek FSI3 case
    if (parameters.scenario == "FSI3")
      {
        // FSI 3
        n_x          = 18;
        n_y          = 3;
        n_z          = 1;
        point_bottom = dim == 3 ? Point<dim>(0.24899, 0.19, -0.005) :
                                  Point<dim>(0.24899, 0.19);
        point_tip =
          dim == 3 ? Point<dim>(0.6, 0.21, 0.005) : Point<dim>(0.6, 0.21);

        // IDs for FSI3
        id_flap_long_bottom  = 2; // x direction
        id_flap_long_top     = 3;
        id_flap_short_bottom = 0; // y direction
        id_flap_short_top    = 1;

        // only relevant for quasi-2D
        id_flap_out_of_plane_bottom = 4; // z direction
        id_flap_out_of_plane_top    = 5;
      }
    // PF Case
    else if (parameters.scenario == "PF")
      {
        n_x = 3;
        n_y = 18;
        n_z = 1;

        double flap_xlocation = parameters.flap_location;

        point_bottom = dim == 3 ? Point<dim>(flap_xlocation - 0.05, 0, 0) :
                                  Point<dim>(flap_xlocation - 0.05, 0);
        point_tip    = dim == 3 ? Point<dim>(flap_xlocation + 0.05, 1, 0.3) :
                                  Point<dim>(flap_xlocation + 0.05,
                                          1); // flap has a 0.1 width

        // IDs for PF
        id_flap_long_bottom  = 0; // x direction
        id_flap_long_top     = 1;
        id_flap_short_bottom = 2; // y direction
        id_flap_short_top    = 3;

        // only relevant for quasi-2D
        id_flap_out_of_plane_bottom = 4; // z direction
        id_flap_out_of_plane_top    = 5;
      }
    // Flying-sled foam airframe: the flat plate of the reference example
    // (test/flying-sled). It runs from the clamped leading edge (x = 0.45)
    // through the elevon hinge line (x = 0.55) to the elevon trailing edge
    // (x = 0.62), so the elevons (x in [0.55, 0.62]) are part of the same
    // structure. In 3D the plate is built by make_sled_grid(); this block is
    // the 2D fallback.
    else if (parameters.scenario == "Sled")
      {
        if (dim == 3)
          {
            make_sled_grid();
            return;
          }
        n_x = 17; // 0.01 m cells, so a cell face lies on the hinge x = 0.55
        n_y = 12;
        n_z = 1;

        const double t = parameters.plate_thickness; // foam plate [m]

        point_bottom = dim == 3 ? Point<dim>(0.45, 0.10, -0.5 * t) :
                                  Point<dim>(0.45, 0.10);
        point_tip    = dim == 3 ? Point<dim>(0.62, 0.40, 0.5 * t) :
                                  Point<dim>(0.62, 0.40);

        // colorized boundary IDs: 0 = x-, 1 = x+, 2 = y-, 3 = y+, 4 = z-,
        // 5 = z+. Couple the upper face (z+) to the fluid and clamp the
        // leading edge (x-); leave the remaining faces free. There is no
        // quasi-2D out-of-plane clamp for the sled plate, so those IDs point
        // at faces that do not exist.
        id_flap_long_bottom         = 7; // unused
        id_flap_long_top            = 7; // unused
        id_flap_short_bottom        = 0; // x- (leading edge) -> clamped
        id_flap_short_top           = 5; // z+ -> coupling interface
        id_flap_out_of_plane_bottom = 8; // unused
        id_flap_out_of_plane_top    = 9; // unused
      }
    else
      {
        AssertThrow(false,
                    ExcMessage("Unknown scenario: " + parameters.scenario));
      }

    // Vector of dim values denoting the number of cells to generate in that
    // direction
    const std::vector<unsigned int> repetitions =
      dim == 2 ? std::vector<unsigned int>({n_x, n_y}) :
                 std::vector<unsigned int>({n_x, n_y, n_z});

    GridGenerator::subdivided_hyper_rectangle(triangulation,
                                              repetitions,
                                              point_bottom,
                                              point_tip,
                                              /*colorize*/ true);

    // Refine all cells global_refinement times
    const unsigned int global_refinement = 0;
    triangulation.refine_global(global_refinement);

    // Set the desired IDs for clamped boundaries and out_of_plane clamped
    // boundaries. The interface ID (refering to the coupling) is specified in
    // the Constructor, since it is needed by the Constructor of the Adapter
    // class.
    clamped_mesh_id              = 0;
    out_of_plane_clamped_mesh_id = id_flap_out_of_plane_bottom;

    // The IDs must not be the same:
    std::string error_message(
      "The interface_id cannot be the same as the clamped one");
    AssertThrow(clamped_mesh_id != interface_boundary_id,
                ExcMessage(error_message));
    AssertThrow(out_of_plane_clamped_mesh_id != interface_boundary_id,
                ExcMessage(error_message));
    AssertThrow(interface_boundary_id == adapter.deal_boundary_interface_id,
                ExcMessage("Wrong interface ID in the Adapter specified"));

    // Iterate over all cells and set the IDs
    for (const auto &cell : triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary() == true)
          {
            // Boundaries for the interface
            if (face->boundary_id() == id_flap_short_top ||
                face->boundary_id() == id_flap_long_bottom ||
                face->boundary_id() == id_flap_long_top)
              face->set_boundary_id(interface_boundary_id);
            // Boundaries clamped in all directions
            else if (face->boundary_id() == id_flap_short_bottom)
              face->set_boundary_id(clamped_mesh_id);
            // Boundaries clamped out-of-plane (z) direction
            else if (face->boundary_id() == id_flap_out_of_plane_bottom ||
                     face->boundary_id() == id_flap_out_of_plane_top)
              face->set_boundary_id(out_of_plane_clamped_mesh_id);
          }
  }



  // The 3D flying-sled plate. With the servo hinge it is built from three
  // pieces that share no vertices: the airframe (x in [0.45, 0.55]) and one
  // flap per elevon (x in [0.55, 0.62], y in [0.10, 0.25] and [0.25, 0.40]).
  // The flaps are joined to the airframe only by the hinge-line ties in
  // setup_servo_hinges(), so each elevon can rotate rigidly about the hinge
  // and the two elevons can deflect differentially. Without the servo it is a
  // single block with a cell face on the hinge line.
  //
  // Material ids: 0 = airframe, 1 + s = flap of surface s. Boundary ids: the
  // leading edge (x = 0.45) is clamped, the upper face (z+) of the airframe is
  // the coupling interface (the fluid airframe patch stops at the hinge; the
  // elevon loads reach the solid through the hinge data), and every other face
  // is free.
  template <int dim>
  void
  ElastoDynamics<dim>::make_sled_grid()
  {
    const double t   = parameters.plate_thickness; // foam plate [m]
    const double xLE = 0.45, xH = 0.55, xTE = 0.62;
    const double y0 = 0.10, yM = 0.25, y1 = 0.40;

    const auto block = [t](Triangulation<dim> &tria,
                           const unsigned int  nx,
                           const unsigned int  ny,
                           const double        xa,
                           const double        xb,
                           const double        ya,
                           const double        yb) {
      GridGenerator::subdivided_hyper_rectangle(
        tria,
        std::vector<unsigned int>({nx, ny, 1}),
        Point<dim>(xa, ya, -0.5 * t),
        Point<dim>(xb, yb, 0.5 * t));
    };

    // About 0.024 m chordwise and 0.025 m spanwise cells in every piece, with
    // a cell face on the hinge line.
    if (servo_enabled)
      {
        Triangulation<dim> airframe, flap_left, flap_right;
        block(airframe, 4, 12, xLE, xH, y0, y1);
        block(flap_left, 3, 6, xH, xTE, y0, yM);
        block(flap_right, 3, 6, xH, xTE, yM, y1);
        // A zero vertex tolerance keeps the coincident hinge and inter-elevon
        // vertices separate.
        GridGenerator::merge_triangulations({&airframe, &flap_left, &flap_right},
                                            triangulation,
                                            /*duplicated_vertex_tolerance*/ 0.0);
        for (const auto &cell : triangulation.active_cell_iterators())
          cell->set_material_id(cell->center()[0] < xH ?
                                  0 :
                                  (cell->center()[1] < yM ? 1 : 2));
      }
    else
      {
        // One continuous plate: the default tolerance merges the vertices on
        // the hinge line.
        Triangulation<dim> airframe, flaps;
        block(airframe, 4, 12, xLE, xH, y0, y1);
        block(flaps, 3, 12, xH, xTE, y0, y1);
        GridGenerator::merge_triangulations(airframe, flaps, triangulation);
        for (const auto &cell : triangulation.active_cell_iterators())
          cell->set_material_id(0);
      }

    clamped_mesh_id              = 0;
    out_of_plane_clamped_mesh_id = 9; // no face carries it
    const types::boundary_id free_id = 7;
    AssertThrow(interface_boundary_id == adapter.deal_boundary_interface_id,
                ExcMessage("Wrong interface ID in the Adapter specified"));
    AssertThrow(interface_boundary_id != clamped_mesh_id &&
                  interface_boundary_id != free_id &&
                  interface_boundary_id != out_of_plane_clamped_mesh_id,
                ExcMessage("Conflicting Sled boundary IDs"));

    const double tol = 1.0e-9;
    for (const auto &cell : triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary())
          {
            const Point<dim> c = face->center();
            if (std::abs(c[0] - xLE) < tol)
              face->set_boundary_id(clamped_mesh_id);
            else if (c[2] > 0.5 * t - tol && c[0] < xH)
              face->set_boundary_id(interface_boundary_id);
            else
              face->set_boundary_id(free_id);
          }
  }



  template <int dim>
  void
  ElastoDynamics<dim>::setup_system()
  {
    // This follows the usual dealii steps
    dof_handler.distribute_dofs(fe);
    hanging_node_constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler,
                                            hanging_node_constraints);
    hanging_node_constraints.close();

    n_elastic_dofs = dof_handler.n_dofs();

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    hanging_node_constraints,
                                    /*keep_constrained_dofs = */ true);
    sparsity_pattern.copy_from(dsp);

    // Initialize relevant matrices. The mass and stiffness matrices stay of
    // elastic size; the stepping/system matrices are augmented to
    // n_elastic_dofs + n_flaps in assemble_system() when servos are enabled.
    mass_matrix.reinit(sparsity_pattern);
    stiffness_matrix.reinit(sparsity_pattern);
    if (!servo_enabled)
      {
        system_matrix.reinit(sparsity_pattern);
        stepping_matrix.reinit(sparsity_pattern);
      }

    // Initialize all vectors. With servos, the time dependent vectors carry
    // one extra theta entry per surface (one flap per surface, checked in
    // setup_servo_hinges()) after the elastic dofs.
    const unsigned int n_aug =
      servo_enabled
        ? n_elastic_dofs +
            parameters.servo_hinge_locations.size() / dim
        : n_elastic_dofs;
    old_velocity.reinit(n_aug);
    velocity.reinit(n_aug);

    old_displacement.reinit(n_aug);
    displacement.reinit(n_aug);

    system_rhs.reinit(n_aug);
    old_stress.reinit(n_aug);
    stress.reinit(n_aug);

    body_force_vector.reinit(n_aug);

    std::cout.imbue(std::locale(""));
    std::cout << "Triangulation:"
              << "\n\t Number of active cells: "
              << triangulation.n_active_cells()
              << "\n\t Polynomial degree: " << parameters.poly_degree
              << "\n\t Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

    // Define alias for time dependent variables as described above
    state_variables = {
      &old_velocity, &velocity, &old_displacement, &displacement, &old_stress};
    // The servo states are time-dependent too; keep them checkpointed so an
    // implicit-coupling re-subiteration reloads them together with the
    // elastic vectors.
    if (servo_enabled)
      {
        state_variables.push_back(&servo_theta);
        state_variables.push_back(&servo_omega);
        state_variables.push_back(&servo_cmd);
      }

    // loads at time 0
    // TODO: Check, if initial conditions should be set at the beginning
    old_stress = 0.0;
  }



  template <int dim>
  void
  ElastoDynamics<dim>::assemble_system()
  {
    QGauss<dim> quadrature_formula(quad_order);

    FEValues<dim> fe_values(mapping,
                            fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> lambda_values(n_q_points);
    std::vector<double> mu_values(n_q_points);

    // Lame constants
    Functions::ConstantFunction<dim> lambda(parameters.lambda),
      mu(parameters.mu);

    // Assemble the stiffness matrix according to a linear material law using
    // the lame paramters
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix = 0;

        fe_values.reinit(cell);

        // Next we get the values of the coefficients at the quadrature
        // points.
        lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
        mu.value_list(fe_values.get_quadrature_points(), mu_values);


        // Then assemble the entries of the local stiffness matrix
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            const unsigned int component_i =
              fe.system_to_component_index(i).first;

            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                const unsigned int component_j =
                  fe.system_to_component_index(j).first;

                for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                  {
                    cell_matrix(i, j) +=
                      // the first term is (lambda d_i u_i, d_j v_j) + (mu d_i
                      // u_j, d_j v_i).
                      (                                                  //
                        (fe_values.shape_grad(i, q_point)[component_i] * //
                         fe_values.shape_grad(j, q_point)[component_j] * //
                         lambda_values[q_point])                         //
                        +                                                //
                        (fe_values.shape_grad(i, q_point)[component_j] * //
                         fe_values.shape_grad(j, q_point)[component_i] * //
                         mu_values[q_point])                             //
                        +                                                //
                        // the second term is (mu nabla u_i, nabla v_j).
                        ((component_i == component_j) ?        //
                           (fe_values.shape_grad(i, q_point) * //
                            fe_values.shape_grad(j, q_point) * //
                            mu_values[q_point]) :              //
                           0)                                  //
                        ) *                                    //
                      fe_values.JxW(q_point);                  //
                  }
              }
          }


        // The transfer from local degrees of freedom into the global matrix
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              stiffness_matrix.add(local_dof_indices[i],
                                   local_dof_indices[j],
                                   cell_matrix(i, j));
          }
      }


    // Here, we use the MatrixCreator to create a mass matrix, which is constant
    // through the whole simulation
    {
      Functions::ConstantFunction<dim> rho_f(parameters.rho);

      MatrixCreator::create_mass_matrix(
        mapping, dof_handler, QGauss<dim>(quad_order), mass_matrix, &rho_f);
    }

    // Then, we save the system_matrix, which is needed every timestep
    if (!servo_enabled)
      {
        stepping_matrix.copy_from(stiffness_matrix);
        stepping_matrix *= time.get_delta_t() * time.get_delta_t() *
                           parameters.theta * parameters.theta;
        stepping_matrix.add(1, mass_matrix);
        hanging_node_constraints.condense(stepping_matrix);
      }
    else
      {
        // Elastic stepping block at elastic size.
        SparseMatrix<double> stepping_elastic;
        stepping_elastic.reinit(sparsity_pattern);
        stepping_elastic.copy_from(stiffness_matrix);
        stepping_elastic *= time.get_delta_t() * time.get_delta_t() *
                            parameters.theta * parameters.theta;
        stepping_elastic.add(1, mass_matrix);

        // Augmented N x N pattern: the elastic block plus one theta row per
        // flap, coupled to its own flap dofs. The gauge/hinge constraints
        // are condensed into the pattern below, which adds exactly the fill-in
        // they require (no dense flap block).
        // DoFTools needs a pattern of exactly n_dofs rows, so build the
        // elastic block at elastic size and copy it in.
        const unsigned int     N = n_elastic_dofs + n_flaps;
        DynamicSparsityPattern dsp_elastic(n_elastic_dofs, n_elastic_dofs);
        DoFTools::make_sparsity_pattern(dof_handler,
                                        dsp_elastic,
                                        servo_constraints,
                                        /*keep_constrained_dofs = */ true);
        DynamicSparsityPattern dsp_aug(N, N);
        for (unsigned int i = 0; i < n_elastic_dofs; ++i)
          for (auto it = dsp_elastic.begin(i); it != dsp_elastic.end(i); ++it)
            dsp_aug.add(i, it->column());
        for (const ServoHinge &sh : servo_hinges)
          for (auto i : sh.flap_dofs)
            {
              dsp_aug.add(sh.dof, i);
              dsp_aug.add(i, sh.dof);
            }
        servo_constraints.condense(dsp_aug);
        augmented_sparsity_pattern.copy_from(dsp_aug);
        system_matrix.reinit(augmented_sparsity_pattern);
        stepping_matrix.reinit(augmented_sparsity_pattern);

        // Copy the elastic block into the augmented matrix and condense the
        // servo constraints (hinge ties + gauge) before the boundary values
        // are applied.
        stepping_matrix = 0.0;
        for (unsigned int i = 0; i < n_elastic_dofs; ++i)
          for (SparseMatrix<double>::const_iterator it =
                 stepping_elastic.begin(i);
               it != stepping_elastic.end(i);
               ++it)
            stepping_matrix.add(i, it->column(), it->value());
        servo_constraints.condense(stepping_matrix);
      }

    // Calculate contribution of gravity and store them in gravitational_force
    if (body_force_enabled)
      {
        Vector<double> bf_vector(dim);
        for (uint d = 0; d < dim; ++d)
          bf_vector[d] = parameters.rho * parameters.body_force[d];

        // Create a constant function object
        Functions::ConstantFunction<dim> bf_function(bf_vector);

        // Create the contribution to the right-hand side vector. VectorTools
        // needs a vector of exactly n_dofs entries; the servo's theta
        // entries get no body force.
        Vector<double> bf_rhs(dof_handler.n_dofs());
        VectorTools::create_right_hand_side(mapping,
                                            dof_handler,
                                            QGauss<dim>(quad_order),
                                            bf_function,
                                            bf_rhs);
        body_force_vector = 0.0;
        for (unsigned int i = 0; i < bf_rhs.size(); ++i)
          body_force_vector[i] = bf_rhs[i];
      }
  }


  // Process RHS assembly, which is the coupling data (stress) in this case
  template <int dim>
  void
  ElastoDynamics<dim>::assemble_rhs()
  {
    timer.enter_subsection("Assemble rhs");

    // In case we get consistent data
    if (parameters.data_consistent)
      assemble_consistent_loading();
    else // In case we get conservative data
      system_rhs = stress;
    // Update time dependent variables related to the previous time step t_n
    old_velocity     = velocity;
    old_displacement = displacement;

    // Add contribution of body forces, if necessary
    if (body_force_enabled)
      system_rhs.add(1, body_force_vector);

    // Add the propeller point load + couple, if enabled
    if (prop_enabled)
      add_propeller_rhs();

    // Add the control-surface hinge point load + couple, if enabled
    if (hinge_enabled)
      add_hinge_rhs();

    // Assemble global RHS:
    // RHS=(M-theta*(1-theta)*delta_t^2*K)*V_n - delta_t*K* D_n +
    // delta_t*theta*F_n+1 + delta_t*(1-theta)*F_n

    // tmp vector to store intermediate results
    Vector<double> tmp;
    tmp.reinit(servo_enabled ? n_elastic_dofs + n_flaps :
                               dof_handler.n_dofs());

    // The mass/stiffness matrices are of elastic size, so in the servo case
    // apply them only to the elastic part of the augmented vectors.
    Vector<double> src_n, dst_n;
    if (servo_enabled)
      {
        src_n.reinit(n_elastic_dofs);
        dst_n.reinit(n_elastic_dofs);
      }
    const auto elastic_vmult = [this, &src_n, &dst_n](
                                 const SparseMatrix<double> &M,
                                 Vector<double> &            dst,
                                 const Vector<double> &      src) {
      if (!servo_enabled)
        {
          M.vmult(dst, src);
          return;
        }
      for (unsigned int i = 0; i < n_elastic_dofs; ++i)
        src_n[i] = src[i];
      M.vmult(dst_n, src_n);
      for (unsigned int i = 0; i < n_elastic_dofs; ++i)
        dst[i] = dst_n[i];
      for (unsigned int i = n_elastic_dofs; i < dst.size(); ++i)
        dst[i] = 0.0;
    };

    tmp = system_rhs;

    system_rhs *= time.get_delta_t() * parameters.theta;
    system_rhs.add(time.get_delta_t() * (1 - parameters.theta), old_stress);
    old_stress = tmp;

    elastic_vmult(mass_matrix, tmp, old_velocity);
    system_rhs.add(1, tmp);

    elastic_vmult(stiffness_matrix, tmp, old_velocity);
    system_rhs.add(-parameters.theta * time.get_delta_t() * time.get_delta_t() *
                     (1 - parameters.theta),
                   tmp);

    elastic_vmult(stiffness_matrix, tmp, old_displacement);
    system_rhs.add(-time.get_delta_t(), tmp);

    if (servo_enabled)
      servo_constraints.condense(system_rhs);
    else
      hanging_node_constraints.condense(system_rhs);

    // Theta rows of the RHS: I_ss * omega keeps the solved theta velocity
    // consistent with the explicit servo state (overwritten by
    // update_servo_hinges() after the solve anyway).
    if (servo_enabled)
      for (unsigned int s = 0; s < n_flaps; ++s)
        system_rhs(servo_hinges[s].dof) = servo_inertia_row[s] * servo_omega[s];

    // Copy the system_matrix every timestep, since applying the BC deletes
    // certain rows and columns
    system_matrix = 0.0;
    system_matrix.copy_from(stepping_matrix);

    // Set Dirichlet BCs:
    // clamped in all directions
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             clamped_mesh_id,
                                             Functions::ZeroFunction<dim>(dim),
                                             boundary_values);
    if (dim == 3)
      {
        const FEValuesExtractors::Scalar z_component(2);
        // clamped out_of_plane
        VectorTools::interpolate_boundary_values(
          dof_handler,
          out_of_plane_clamped_mesh_id,
          Functions::ZeroFunction<dim>(dim),
          boundary_values,
          fe.component_mask(z_component));
      }

    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       velocity,
                                       system_rhs);

    timer.leave_subsection("Assemble rhs");
  }



  // ---------------------------------------------------------------------
  // Actuated control-surface hinge (rotational DOF + servo)
  // ---------------------------------------------------------------------
  //
  // The control surface is part of the same elastic plate as the airframe. The
  // mesh is split at the hinge into an airframe block and a flap block whose
  // hinge-line nodes are duplicated. The flap keeps its elastic bending; its
  // rigid rotation about the hinge is carried by one extra scalar DOF `theta`,
  // driven by the servo model. The flap hinge nodes are tied to the airframe,
  // and a mass-orthogonal gauge (g^T M_flap u_el = 0) removes the flap's own
  // rigid rotation from the elastic DOFs, so the system stays SPD and the
  // existing Direct solver works unchanged.

  namespace
  {
    // g(x) = axis x (x - hinge), the rigid-rotation displacement field of the
    // flap about the hinge line.
    template <int dim>
    inline Point<dim>
    hinge_g(const Point<dim> &axis, const Point<dim> &r)
    {
      Point<dim> g;
      if (dim == 3)
        g = Point<dim>(axis[1] * r[2] - axis[2] * r[1],
                       axis[2] * r[0] - axis[0] * r[2],
                       axis[0] * r[1] - axis[1] * r[0]);
      else
        g = Point<dim>(-axis[1] * r[1], axis[0] * r[0]);
      return g;
    }
  } // namespace

  template <int dim>
  void
  ElastoDynamics<dim>::setup_servo_hinges()
  {
    if (!servo_enabled)
      return;

    AssertThrow(dim == 3 && parameters.scenario == "Sled",
                ExcMessage("The servo hinge needs the 3D Sled plate, which "
                           "builds one mesh piece per flap."));

    n_elastic_dofs = dof_handler.n_dofs();
    const unsigned int n_surf =
      parameters.servo_hinge_locations.size() / dim;
    AssertThrow(parameters.servo_hinge_locations.size() % dim == 0 && n_surf > 0,
                ExcMessage("Servo hinge locations must be a whole number of "
                           "dim-tuples."));

    Point<dim> axis;
    for (unsigned int d = 0; d < dim; ++d)
      axis[d] = parameters.hinge_axis[d];
    AssertThrow(axis.norm() > 1e-12, ExcMessage("Hinge axis must be nonzero."));
    axis /= axis.norm();

    std::map<types::global_dof_index, Point<dim>> support_points;
    DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);
    // Component of each global dof, built from cell-local lookups
    // (system_to_component_index() only accepts cell-local indices).
    std::map<types::global_dof_index, unsigned int> dof_component;
    {
      std::vector<types::global_dof_index> local_dofs(fe.dofs_per_cell);
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          cell->get_dof_indices(local_dofs);
          for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
            dof_component[local_dofs[i]] =
              fe.system_to_component_index(i).first;
        }
    }
    const auto comp = [&dof_component](const types::global_dof_index i) {
      return dof_component.at(i);
    };

    // Surface chord direction: in the plate plane (perpendicular to the plate
    // normal z) and perpendicular to the hinge axis. For the flat-plate sled
    // (axis = +y) this gives +x.
    Point<dim> normal;
    normal[dim - 1] = 1.0;
    Point<dim> chord;
    if (dim == 3)
      chord = Point<dim>(axis[1] * normal[2] - axis[2] * normal[1],
                         axis[2] * normal[0] - axis[0] * normal[2],
                         axis[0] * normal[1] - axis[1] * normal[0]);
    else
      {
        chord[0] = -axis[1];
        chord[1] = axis[0];
      }
    AssertThrow(chord.norm() > 1e-12,
                ExcMessage("Could not determine the surface chord."));
    chord /= chord.norm();

    // The flap pieces (material id > 0, see make_sled_grid()) and their span
    // along the hinge axis. Each control surface actuates exactly one of
    // them: the one whose span contains its hinge point.
    std::map<types::material_id, std::pair<double, double>> flap_span;
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->material_id() != 0)
        {
          auto it = flap_span
                      .emplace(cell->material_id(),
                               std::make_pair(std::numeric_limits<double>::max(),
                                              -std::numeric_limits<double>::max()))
                      .first;
          for (const unsigned int v : cell->vertex_indices())
            {
              const double a = cell->vertex(v) * axis;
              it->second.first  = std::min(it->second.first, a);
              it->second.second = std::max(it->second.second, a);
            }
        }
    AssertThrow(flap_span.size() == n_surf,
                ExcMessage("Servo hinge locations: expected one hinge per flap (" +
                           std::to_string(flap_span.size()) + "), got " +
                           std::to_string(n_surf) + "."));

    servo_command_values.assign(n_surf, 0.0);
    servo_hinge_moment_axis.assign(n_surf, 0.0);
    servo_constraints.clear();
    servo_constraints.merge(hanging_node_constraints);

    // Airframe dofs (the tie targets).
    std::set<types::global_dof_index> air_dofs;
    {
      std::vector<types::global_dof_index> ld(fe.dofs_per_cell);
      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->material_id() == 0)
          {
            cell->get_dof_indices(ld);
            air_dofs.insert(ld.begin(), ld.end());
          }
    }

    std::set<types::material_id>      used_flaps;
    std::set<types::global_dof_index> all_flap_dofs;
    servo_hinges.assign(n_surf, ServoHinge());
    for (unsigned int s = 0; s < n_surf; ++s)
      {
        ServoHinge &sh = servo_hinges[s];
        for (unsigned int d = 0; d < dim; ++d)
          sh.hinge_point[d] = parameters.servo_hinge_locations[s * dim + d];
        sh.axis = axis;
        sh.dof  = n_elastic_dofs + s;

        const double a_h     = sh.hinge_point * axis;
        types::material_id m = numbers::invalid_material_id;
        for (const auto &e : flap_span)
          if (a_h > e.second.first + 1.0e-9 && a_h < e.second.second - 1.0e-9)
            m = e.first;
        AssertThrow(m != numbers::invalid_material_id,
                    ExcMessage("Servo hinge location " + std::to_string(s) +
                               " is not within the span of any flap."));
        AssertThrow(used_flaps.insert(m).second,
                    ExcMessage("Two servo hinge locations lie on the same "
                               "flap; give one per flap."));

        std::set<types::global_dof_index> flap_dofs;
        {
          std::vector<types::global_dof_index> ld(fe.dofs_per_cell);
          for (const auto &cell : dof_handler.active_cell_iterators())
            if (cell->material_id() == m)
              {
                cell->get_dof_indices(ld);
                flap_dofs.insert(ld.begin(), ld.end());
              }
        }
        // The flap must be a separate mesh piece: sharing a vertex with the
        // airframe or the other flap would couple its rigid rotation to them.
        for (auto i : flap_dofs)
          AssertThrow(air_dofs.count(i) == 0 && all_flap_dofs.insert(i).second,
                      ExcMessage("Servo flap " + std::to_string(s) +
                                 " shares dofs with the airframe or another "
                                 "flap; the mesh pieces were merged."));

        // Coupling vector and rotary inertia of the flap about the hinge:
        //   g(x) = axis x (x - hinge)
        //   cg[j] = int rho phi_j . g dV,  I = int rho g.g dV
        const QGauss<dim> quad(quad_order);
        FEValues<dim>     fe_values(mapping,
                                fe,
                                quad,
                                update_values | update_quadrature_points |
                                  update_JxW_values);
        std::vector<types::global_dof_index> ld(fe.dofs_per_cell);
        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            if (cell->material_id() != m)
              continue;
            fe_values.reinit(cell);
            cell->get_dof_indices(ld);
            for (unsigned int q = 0; q < quad.size(); ++q)
              {
                const Point<dim> &X = fe_values.quadrature_point(q);
                const double      w = fe_values.JxW(q);
                Point<dim>        r;
                r = X;
                r -= sh.hinge_point;
                const Point<dim> g = hinge_g<dim>(axis, r);
                sh.inertia += parameters.rho * (g * g) * w;
                for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
                  sh.cg[ld[i]] += parameters.rho * fe_values.shape_value(i, q) *
                                  g[comp(ld[i])] * w;
              }
          }

        // Hinge tie: tie each flap node on the hinge line to the co-located
        // airframe dof of the same component. The flap is a separate mesh
        // piece, so these are its only connection to the airframe; the flap
        // can then rotate rigidly about the hinge line (the theta mode, which
        // the gauge below removes from the elastic dofs).
        unsigned int n_ties = 0;
        for (auto f : flap_dofs)
          {
            const Point<dim>  X = support_points[f];
            Point<dim>        r = X;
            r -= sh.hinge_point;
            const Tensor<1, dim> proj = r - (r * sh.axis) * sh.axis;
            if (proj.norm() > 1.0e-6)
              continue;
            for (auto a : air_dofs)
              if (comp(a) == comp(f) &&
                  (support_points[a] - X).norm() < 1.0e-6)
                {
                  servo_constraints.add_line(f);
                  servo_constraints.add_entry(f, a, 1.0);
                  ++n_ties;
                  break;
                }
          }
        AssertThrow(n_ties > 0,
                    ExcMessage("Servo hinge " + std::to_string(s) +
                               ": no flap node lies on the hinge line. The "
                               "hinge point must lie on the flap's hinge edge "
                               "(for the Sled plate x = 0.55, z = 0)."));

        // Gauge: g^T M_flap u_el = 0 removes the flap's rigid rotation from
        // the elastic dofs. Pick the flap dof with the largest |cg[j]| as
        // slave and eliminate it.
        types::global_dof_index slave = numbers::invalid_dof_index;
        double                  cgmax = 0.0;
        for (const auto &e : sh.cg)
          if (!servo_constraints.is_constrained(e.first) &&
              std::abs(e.second) > cgmax)
            {
              cgmax = std::abs(e.second);
              slave = e.first;
            }
        AssertThrow(slave != numbers::invalid_dof_index,
                    ExcMessage("Could not select a gauge slave dof for servo "
                               "hinge " +
                               std::to_string(s)));
        servo_constraints.add_line(slave);
        for (const auto &e : sh.cg)
          if (e.first != slave)
            servo_constraints.add_entry(slave, e.first,
                                        -e.second / sh.cg[slave]);

        for (auto i : flap_dofs)
          {
            sh.flap_dofs.push_back(i);
            sh.flap_points.push_back(support_points[i]);
            sh.flap_components.push_back(dof_component.at(i));
          }
      }
    n_flaps = n_surf;
    servo_theta.reinit(n_flaps);
    servo_omega.reinit(n_flaps);
    servo_cmd.reinit(n_flaps);
    servo_constraints.close();

    // setup_system() sized the time-dependent vectors for one theta per
    // surface; with one flap per surface that is one theta per flap.
    AssertThrow(displacement.size() == n_elastic_dofs + n_flaps &&
                  system_rhs.size() == n_elastic_dofs + n_flaps,
                ExcMessage("Servo: augmented vector size mismatch."));
  }



  template <int dim>
  void
  ElastoDynamics<dim>::add_servo_terms()
  {
    if (!servo_enabled)
      return;

    // The theta row carries the flap + reflected servo inertia, so the solve
    // reproduces the explicit servo rate (see assemble_rhs()).
    servo_inertia_row.assign(n_flaps, 0.0);
    for (unsigned int s = 0; s < n_flaps; ++s)
      {
        const ServoHinge &sh = servo_hinges[s];
        servo_inertia_row[s] = sh.inertia + parameters.servo_inertia;
        stepping_matrix.add(sh.dof, sh.dof, servo_inertia_row[s]);
      }
  }



  template <int dim>
  void
  ElastoDynamics<dim>::update_servo_hinges()
  {
    if (!servo_enabled)
      return;

    const double dt = time.get_delta_t();

    for (unsigned int s = 0; s < n_flaps; ++s)
      {
        const ServoHinge &sh = servo_hinges[s];

        // Reload the servo state from the checkpointed vectors. For an implicit
        // coupling this reverts theta/omega/cmd to the start of the window
        // before the step is repeated.
        rb::ServoModel servo;
        servo.kp        = parameters.servo_kp;
        servo.kd        = parameters.servo_kd;
        servo.torqueMax = parameters.servo_torque_max;
        servo.rateMax   = parameters.servo_rate_max;
        servo.freeplay  = parameters.servo_freeplay;
        servo.coulomb   = parameters.servo_coulomb;
        servo.viscous   = parameters.servo_viscous;
        servo.inertia   = sh.inertia + parameters.servo_inertia;
        servo.theta     = servo_theta[s];
        servo.omega     = servo_omega[s];
        servo.cmd       = servo_cmd[s];

        const double theta_before = servo.theta;

        rb::servoStep(servo,
                      s < servo_command_values.size() ?
                        servo_command_values[s] :
                        0.0,
                      s < servo_hinge_moment_axis.size() ?
                        servo_hinge_moment_axis[s] :
                        0.0,
                      dt);

        const double theta = servo.theta;
        const double omega = servo.omega;
        const double cmd   = servo.cmd;

        // Write the rigid rotation theta back into the elastic displacement of
        // the flap. displacement[j] already carries u_el (bending only, the
        // gauge keeps the rigid mode out of the elastic solution); adding the
        // delta keeps the total = u_el + theta*g without double counting.
        const double dtheta = theta - theta_before;
        for (std::size_t k = 0; k < sh.flap_dofs.size(); ++k)
          {
            const types::global_dof_index j = sh.flap_dofs[k];
            Point<dim>                    r = sh.flap_points[k];
            r -= sh.hinge_point;
            const Point<dim> g = hinge_g<dim>(sh.axis, r);
            displacement[j] += dtheta * g[sh.flap_components[k]];
          }
        displacement[sh.dof] = theta;
        velocity[sh.dof]     = omega;

        // Publish the advanced state back into the checkpointed vectors.
        servo_theta[s] = theta;
        servo_omega[s] = omega;
        servo_cmd[s]   = cmd;

        // Machine-parseable servo state, one line per surface per time step
        // (consumed by tests/servo_fem_smoke.py).
        // std::cout carries the user's locale; format this line in the
        // classic one so a decimal comma never reaches the parser.
        std::ostringstream servo_line;
        servo_line.copyfmt(std::cout);
        servo_line.imbue(std::locale::classic());
        servo_line << "SERVO s=" << s << " theta=" << theta
                   << " omega=" << omega;
        std::cout << servo_line.str() << std::endl;
      }
  }


  // Process RHS assembly, which is the coupling data (stress) in this case
  template <int dim>
  void
  ElastoDynamics<dim>::assemble_consistent_loading()
  { // Initialize all objects as usual
    system_rhs = 0.0;

    // Quadrature formula for integration over faces (dim-1)
    QGauss<dim - 1> face_quadrature_formula(quad_order);

    FEFaceValues<dim> fe_face_values(mapping,
                                     fe,
                                     face_quadrature_formula,
                                     update_values | update_JxW_values);

    const unsigned int dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    // FEFaceValues needs a vector of exactly n_dofs entries; with the servo
    // the stress vector also carries the theta entries, so use its elastic
    // part.
    Vector<double> stress_elastic;
    if (servo_enabled)
      {
        stress_elastic.reinit(dof_handler.n_dofs());
        for (unsigned int i = 0; i < stress_elastic.size(); ++i)
          stress_elastic[i] = stress[i];
      }
    const Vector<double> &stress_field = servo_enabled ? stress_elastic : stress;

    Vector<double>                       cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


    // In order to get the local fe values
    std::vector<Vector<double>> local_stress(n_face_q_points,
                                             Vector<double>(dim));

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_rhs = 0;

        // Assemble the right-hand side force vector each timestep
        // by applying contributions only on the coupling interface
        for (const auto &face : cell->face_iterators())
          if (face->at_boundary() == true &&
              face->boundary_id() == interface_boundary_id)
            {
              fe_face_values.reinit(cell, face);
              // Extract relevant data from the global stress vector by using
              // 'get_function_values()'
              // In contrast to the nonlinear solver, no pull back is performed.
              // The equilibrium is stated in reference configuration, but only
              // valid for very small deformations
              fe_face_values.get_function_values(stress_field, local_stress);

              for (unsigned int f_q_point = 0; f_q_point < n_face_q_points;
                   ++f_q_point)
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    const unsigned int component_i =
                      fe.system_to_component_index(i).first;

                    cell_rhs(i) += fe_face_values.shape_value(i, f_q_point) *
                                   local_stress[f_q_point][component_i] *
                                   fe_face_values.JxW(f_q_point);
                  }
            }

        // Local dofs to global
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
          }
      }
  }



  // Apply a point force (vector-valued) at the given point by distributing it
  // to the DoFs of the cell containing the point.
  template <int dim>
  void
  ElastoDynamics<dim>::add_point_force(Vector<double> &     rhs,
                                       const Point<dim> &   point,
                                       const Vector<double> &force)
  {
    bool applied = false;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        const Point<dim> unit_cell_point =
          mapping.transform_real_to_unit_cell(cell, point);

        // Check whether the point lies inside or on the reference cell
        bool inside = true;
        for (unsigned int d = 0; d < dim; ++d)
          if (unit_cell_point[d] < -1.0e-6 ||
              unit_cell_point[d] > 1.0 + 1.0e-6)
            inside = false;

        if (!inside)
          continue;

        std::vector<types::global_dof_index> local_dof_indices(
          fe.dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);

        for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
          {
            const unsigned int component_i =
              fe.system_to_component_index(i).first;
            const double phi_i = fe.shape_value(i, unit_cell_point);

            if (std::abs(phi_i) > 1.0e-14)
              rhs(local_dof_indices[i]) += phi_i * force[component_i];
          }

        applied = true;
        break;
      }

    if (!applied)
      std::cerr << "Propeller point source outside the solid mesh (skipped): "
                << point << std::endl;
  }



  // Apply the propeller loads (one thrust + one torque per hub) as a
  // Point load (thrust) and a couple (torque) about each hub.
  template <int dim>
  void
  ElastoDynamics<dim>::add_propeller_rhs()
  {
    const unsigned int n_prop = adapter.get_n_prop_vertices();
    if (n_prop == 0 || prop_force_values.empty())
      return;

    const std::vector<double> &coords = adapter.get_prop_vertices_coords();

    for (unsigned int hub = 0; hub < n_prop; ++hub)
      {
        Point<dim> hub_point;
        for (unsigned int d = 0; d < dim; ++d)
          hub_point[d] = coords[hub * dim + d];

        Vector<double> thrust(dim);
        Vector<double> torque(dim);
        for (unsigned int d = 0; d < dim; ++d)
          {
            thrust[d] = prop_force_values[hub * dim + d];
            torque[d] = prop_torque_values[hub * dim + d];
          }

        // 1) Point thrust at the hub
        add_point_force(system_rhs, hub_point, thrust);

        // 2) Torque as a couple: for each local axis t, apply two opposing
        //    forces +/- 0.5*(torque x t)/h at hub +/- (h/2) t.
        double h = 0.0;
        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            const Point<dim> uc =
              mapping.transform_real_to_unit_cell(cell, hub_point);
            bool inside = true;
            for (unsigned int d = 0; d < dim; ++d)
              if (uc[d] < -1.0e-6 || uc[d] > 1.0 + 1.0e-6)
                inside = false;
            if (inside)
              {
                h = cell->diameter();
                break;
              }
          }
        if (h <= 0.0)
          {
            std::cerr << "Propeller hub outside the solid mesh; torque skipped."
                      << std::endl;
            continue;
          }
        // Clamp the couple arm so the two couple points stay inside the
        // (possibly thin) structure. The net moment is independent of h.
        h = std::min(h, 0.004);
        const double half = 0.5 * h;

        for (unsigned int axis = 0; axis < dim; ++axis)
          {
            Point<dim> axis_dir;
            axis_dir[axis] = 1.0;

            Vector<double> f(dim);
            for (unsigned int d = 0; d < dim; ++d)
              {
                // 0.5 * (torque x e_axis) / h  (sum over all three axes gives
                // the full torque, see mechanics of a three-axis couple)
                f[d] =
                  0.5 *
                  (torque[(axis + 1) % dim] * axis_dir[(axis + 2) % dim] -
                   torque[(axis + 2) % dim] * axis_dir[(axis + 1) % dim]) /
                  h;
              }

            Point<dim> p_plus  = hub_point;
            Point<dim> p_minus = hub_point;
            p_plus[axis] += half;
            p_minus[axis] -= half;

            Vector<double> f_neg(dim);
            for (unsigned int d = 0; d < dim; ++d)
              f_neg[d] = -f[d];

            add_point_force(system_rhs, p_plus, f);
            add_point_force(system_rhs, p_minus, f_neg);
          }
      }
  }



  // Apply the control-surface hinge loads (one force + one moment per hinge)
  // as a point load (force) and a couple (moment) about each hinge vertex.
  template <int dim>
  void
  ElastoDynamics<dim>::add_hinge_rhs()
  {
    const unsigned int n_hinges = adapter.get_n_hinge_vertices();
    if (n_hinges == 0 || hinge_force_values.empty())
      return;

    const std::vector<double> &coords = adapter.get_hinge_vertices_coords();

    for (unsigned int h = 0; h < n_hinges; ++h)
      {
        Point<dim> hinge_point;
        for (unsigned int d = 0; d < dim; ++d)
          hinge_point[d] = coords[h * dim + d];

        Vector<double> force(dim);
        Vector<double> moment(dim);
        for (unsigned int d = 0; d < dim; ++d)
          {
            force[d]  = hinge_force_values[h * dim + d];
            moment[d] = hinge_moment_values[h * dim + d];
          }

        // With the servo hinge the moment about the hinge axis turns theta
        // (update_servo_hinges()), so applying it here as well would count it
        // twice. The hinge force goes into the structure at the solid-frame
        // hinge point, on the hinge line that ties the flap to the airframe.
        if (servo_enabled)
          {
            if (h < servo_hinges.size())
              add_point_force(system_rhs, servo_hinges[h].hinge_point, force);
            continue;
          }

        // 1) Point force at the hinge
        add_point_force(system_rhs, hinge_point, force);

        // 2) Moment as a three-axis couple (same as the propeller torque)
        double h_arm = 0.0;
        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            const Point<dim> uc =
              mapping.transform_real_to_unit_cell(cell, hinge_point);
            bool inside = true;
            for (unsigned int d = 0; d < dim; ++d)
              if (uc[d] < -1.0e-6 || uc[d] > 1.0 + 1.0e-6)
                inside = false;
            if (inside)
              {
                h_arm = cell->diameter();
                break;
              }
          }
        if (h_arm <= 0.0)
          {
            std::cerr << "Hinge outside the solid mesh; moment skipped."
                      << std::endl;
            continue;
          }
        // Clamp the couple arm so the two couple points stay inside the
        // (possibly thin) structure. The net moment is independent of h.
        h_arm = std::min(h_arm, 0.004);
        const double half = 0.5 * h_arm;

        for (unsigned int axis = 0; axis < dim; ++axis)
          {
            Point<dim> axis_dir;
            axis_dir[axis] = 1.0;

            Vector<double> f(dim);
            for (unsigned int d = 0; d < dim; ++d)
              {
                f[d] =
                  0.5 *
                  (moment[(axis + 1) % dim] * axis_dir[(axis + 2) % dim] -
                   moment[(axis + 2) % dim] * axis_dir[(axis + 1) % dim]) /
                  h_arm;
              }

            Point<dim> p_plus  = hinge_point;
            Point<dim> p_minus = hinge_point;
            p_plus[axis] += half;
            p_minus[axis] -= half;

            Vector<double> f_neg(dim);
            for (unsigned int d = 0; d < dim; ++d)
              f_neg[d] = -f[d];

            add_point_force(system_rhs, p_plus, f);
            add_point_force(system_rhs, p_minus, f_neg);
          }
      }
  }



  template <int dim>
  void
  ElastoDynamics<dim>::solve()
  {
    timer.enter_subsection("Solve system");

    uint   lin_it  = 1;
    double lin_res = 0.0;

    // Solve the linear system either using an iterative CG solver with SSOR or
    // a direct solver using UMFPACK. With servos the gauge-coupled system is
    // solved directly.
    if (parameters.type_lin == "CG" && !servo_enabled)
      {
        std::cout << "\t CG solver: " << std::endl;

        const int solver_its =
          system_matrix.m() * parameters.max_iterations_lin;
        const double tol_sol = 1.e-10;

        SolverControl         solver_control(solver_its, tol_sol);
        GrowingVectorMemory<> GVM;
        SolverCG<>            solver_CG(solver_control, GVM);

        PreconditionSSOR<> preconditioner;
        preconditioner.initialize(system_matrix, 1.2);

        solver_CG.solve(system_matrix, velocity, system_rhs, preconditioner);

        lin_it  = solver_control.last_step();
        lin_res = solver_control.last_value();
      }
    else if (parameters.type_lin == "Direct")
      {
        std::cout << "\t Direct solver: " << std::endl;

        SparseDirectUMFPACK A_direct;
        A_direct.initialize(system_matrix);
        A_direct.vmult(velocity, system_rhs);
      }
    else
      Assert(parameters.type_lin == "Direct" || parameters.type_lin == "CG",
             ExcNotImplemented());

    // assert divergence
    Assert(velocity.linfty_norm() < 1e4, ExcMessage("Linear system diverged"));
    std::cout << "\t     No of iterations:\t" << lin_it
              << "\n \t     Final residual:\t" << lin_res << std::endl;
    if (servo_enabled)
      servo_constraints.distribute(velocity);
    else
      hanging_node_constraints.distribute(velocity);

    timer.leave_subsection("Solve system");
  }



  template <int dim>
  void
  ElastoDynamics<dim>::update_displacement()
  {
    // D_n+1= D_n + delta_t*theta* V_n+1 + delta_t*(1-theta)* V_n
    displacement.add(time.get_delta_t() * parameters.theta, velocity);
    displacement.add(time.get_delta_t() * (1 - parameters.theta), old_velocity);
  }



  template <int dim>
  void
  ElastoDynamics<dim>::output_results() const
  {
    timer.enter_subsection("Output results");

    // The augmented vectors carry theta entries beyond the elastic dofs;
    // output needs the elastic part only.
    Vector<double> displacement_out;
    if (servo_enabled)
      {
        displacement_out.reinit(n_elastic_dofs);
        for (unsigned int i = 0; i < n_elastic_dofs; ++i)
          displacement_out[i] = displacement[i];
      }
    else
      displacement_out = displacement;

    // Declared before data_out, which keeps pointers to both, so that
    // data_out is destroyed first.
    Postprocessor<dim> postprocessor;
    DataOut<dim>       data_out;

    // Note: There is at least paraView v 5.5 needed to visualize this output
    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);

    data_out.attach_dof_handler(dof_handler);

    // The postprocessor class computes straines and passes the displacement to
    // the output
    data_out.add_data_vector(displacement_out, postprocessor);

    // visualize the displacements on a displaced grid
    MappingQEulerian<dim> q_mapping(parameters.poly_degree,
                                    dof_handler,
                                    displacement_out);
    data_out.build_patches(q_mapping,
                           parameters.poly_degree,
                           DataOut<dim>::curved_boundary);

    std::ofstream output(
      parameters.output_folder + "/solution-" +
      Utilities::int_to_string(time.get_timestep() / parameters.output_interval,
                               3) +
      ".vtk");
    data_out.write_vtk(output);
    std::cout << "\t Output written to solution-" +
                   Utilities::int_to_string(time.get_timestep() /
                                              parameters.output_interval,
                                            3) +
                   ".vtk \n"
              << std::endl;
    timer.leave_subsection("Output results");
  }



  template <int dim>
  void
  ElastoDynamics<dim>::run()
  {
    // In the beginning, we create the mesh and set up the data structures
    make_grid();
    setup_system();
    // The servo hinge machinery needs the distributed dofs; it must be in
    // place before the stepping matrix is assembled and condensed.
    if (servo_enabled)
      setup_servo_hinges();
    // A restart continues from a checkpoint; preCICE then gets its
    // displacement as the initial data (if the coupling asks for it).
    if (!parameters.restart_file.empty())
      read_checkpoint(parameters.restart_file);
    output_results();
    assemble_system();
    if (servo_enabled)
      add_servo_terms();

    // Then, we initialize preCICE i.e. we pass our mesh and coupling
    // information to preCICE
    // We aways read data at the end of a time-step, as we blend the beginning
    // and the end via the theta scheme
    if (parameters.prop_enabled)
      {
        adapter.configure_propeller(parameters.prop_mesh_name,
                                    parameters.prop_force_data_name,
                                    parameters.prop_torque_data_name);
        adapter.set_prop_mesh_access_region();
      }
    if (parameters.hinge_enabled || servo_enabled)
      {
        adapter.configure_hinges(parameters.hinge_mesh_name,
                                 parameters.hinge_force_data_name,
                                 parameters.hinge_moment_data_name);
        adapter.set_hinge_mesh_access_region();
      }
    if (servo_enabled)
      {
        adapter.configure_servo_command(parameters.servo_command_data_name);
        adapter.configure_servo_angle(parameters.servo_angle_data_name);
      }
    adapter.initialize(dof_handler, displacement);
    if (parameters.prop_enabled)
      adapter.initialize_prop_mesh();
    if (parameters.hinge_enabled || servo_enabled)
      adapter.initialize_hinge_mesh();
    // The hinge mesh vertices are matched to the servo hinges by index. The
    // fluid-side hinge points need not coincide with the solid-frame ones,
    // but both must list the surfaces in the same spanwise order.
    if (servo_enabled)
      {
        AssertThrow(adapter.get_n_hinge_vertices() == servo_hinges.size(),
                    ExcMessage("The hinge mesh must carry one vertex per servo "
                               "hinge, in the order of \"Servo hinge "
                               "locations\"."));
        const std::vector<double> &coords = adapter.get_hinge_vertices_coords();
        const auto span_order = [this](const auto &span_of) {
          std::vector<unsigned int> order(servo_hinges.size());
          std::iota(order.begin(), order.end(), 0u);
          std::sort(order.begin(),
                    order.end(),
                    [&span_of](unsigned int a, unsigned int b) {
                      return span_of(a) < span_of(b);
                    });
          for (std::size_t k = 1; k < order.size(); ++k)
            AssertThrow(span_of(order[k]) - span_of(order[k - 1]) > 1.0e-9,
                        ExcMessage("Two servo hinges share a spanwise "
                                   "position; their order is ambiguous."));
          return order;
        };
        const auto received = span_order([&](unsigned int s) {
          double a = 0.0;
          for (unsigned int d = 0; d < dim; ++d)
            a += coords[s * dim + d] * servo_hinges[s].axis[d];
          return a;
        });
        const auto own = span_order([this](unsigned int s) {
          return servo_hinges[s].hinge_point * servo_hinges[s].axis;
        });
        AssertThrow(received == own,
                    ExcMessage("The hinge mesh vertices are not in the "
                               "spanwise order of \"Servo hinge locations\"."));
      }

    // Then, we start the time loop. The loop itself is steered by preCICE. This
    // line replaces the usual 'while( time < end_time)'
    while (adapter.precice.isCouplingOngoing())
      {
        // In case of an implicit coupling, we need to store time dependent
        // data, in order to reload it later. The decision, whether it is
        // necessary to store the data is handled by preCICE as well
        adapter.save_current_state_if_required(state_variables, time);

        // Afterwards, we start the actual time step computation
        time.increment();

        std::cout << std::endl
                  << "Timestep " << time.get_timestep() << " @ " << std::fixed
                  << time.current() << "s" << std::endl;

        AssertThrow(std::abs(time.get_delta_t() -
                             adapter.precice.getMaxTimeStepSize()) < 1e-10,
                    ExcMessage(
                      "This solver supports only constant time-step sizes."
                      "Configured time step size in deal.II parameter file: " +
                      std::to_string(time.get_delta_t()) +
                      ". Time-window size from preCICE: " +
                      std::to_string(adapter.precice.getMaxTimeStepSize()) +
                      "."));

        adapter.read_data(time.get_delta_t(), stress);

        // Read the propeller loads (thrust + torque) from the fluid
        if (parameters.prop_enabled)
          adapter.read_prop_data(time.get_delta_t(),
                                 prop_force_values,
                                 prop_torque_values);

        // Read the control-surface hinge loads from the fluid
        if (parameters.hinge_enabled || servo_enabled)
          adapter.read_hinge_data(time.get_delta_t(),
                                  hinge_force_values,
                                  hinge_moment_values);

        // Read the servo command and project the aero hinge moment onto the
        // hinge axis of each surface
        if (servo_enabled)
          {
            adapter.read_servo_command(time.get_delta_t(),
                                       servo_command_values);
            for (std::size_t s = 0; s < servo_hinges.size(); ++s)
              {
                double m_axis = 0.0;
                if ((s + 1) * dim <= hinge_moment_values.size())
                  {
                    Point<dim> m;
                    for (unsigned int d = 0; d < dim; ++d)
                      m[d] = hinge_moment_values[s * dim + d];
                    m_axis = m * servo_hinges[s].axis;
                  }
                servo_hinge_moment_axis[s] = m_axis;
              }
          }

        // Assemble the time dependent contribution obtained from the Fluid
        // participant
        assemble_rhs();

        // ...and solver the system
        solve();

        // Update time dependent data according to the theta-scheme
        update_displacement();

        // Advance the servos explicitly and write the rigid flap rotation back
        // into the displacement before it is handed to the fluid
        if (servo_enabled)
          update_servo_hinges();

        // Then, we exchange data with other participants. Most of the work is
        // done in the adapter: We just need to pass both data vectors with
        // coupling data to the adapter. In case of FSI, 'displacement' is the
        // data we calculate and pass to preCICE and 'stress' is the (global)
        // vector filled by preCICE/ the Fluid participant.
        // Depending on the coupling scheme, we need to wait here for other
        // participant to finish their time step. Therefore, we measure the
        // timings around this functionality
        // Hand the angle the servo actually reached back to the fluid
        if (servo_enabled)
          {
            std::vector<double> angle(servo_theta.begin(), servo_theta.end());
            adapter.write_servo_angle(angle);
          }

        // The interface velocity, for the fluid's predictor: velocity holds
        // the flaps' bending only (update_servo_hinges() adds their rigid
        // rotation to the displacement), so add the rotation rate omega*g.
        if (adapter.writes_velocity())
          {
            Vector<double> coupling_velocity = velocity;
            if (servo_enabled)
              for (unsigned int s = 0; s < n_flaps; ++s)
                {
                  const ServoHinge &sh = servo_hinges[s];
                  for (std::size_t k = 0; k < sh.flap_dofs.size(); ++k)
                    {
                      Point<dim> r = sh.flap_points[k];
                      r -= sh.hinge_point;
                      const Point<dim> g = hinge_g<dim>(sh.axis, r);
                      coupling_velocity[sh.flap_dofs[k]] +=
                        servo_omega[s] * g[sh.flap_components[k]];
                    }
                }
            adapter.write_velocity(coupling_velocity);
          }

        timer.enter_subsection("Advance adapter");
        adapter.advance(displacement, time.get_delta_t());
        timer.leave_subsection("Advance adapter");

        // Next, we reload the data we have previosuly stored in the beginning
        // of the time loop. This is only relevant for implicit couplings and
        // preCICE steeres the reloading depending on the specific
        // configuration.
        adapter.reload_old_state_if_required(state_variables, time);

        // At last, we ask preCICE, whether this coupling time step (= time
        // window in preCICE terms) is finished and write the result files
        if (adapter.precice.isTimeWindowComplete() &&
            time.get_timestep() % parameters.output_interval == 0)
          output_results();

        if (adapter.precice.isTimeWindowComplete() &&
            parameters.checkpoint_interval > 0)
          {
            const double k =
              std::round(time.current() / parameters.checkpoint_interval);
            if (k > 0 &&
                std::abs(time.current() - k * parameters.checkpoint_interval) <
                  0.25 * time.get_delta_t())
              write_checkpoint();
          }
      }

    // After the time loop, we finalize the coupling i.e. terminate
    // communication etc.
    adapter.precice.finalize();
  }

  template <int dim>
  void
  ElastoDynamics<dim>::write_checkpoint() const
  {
    std::ostringstream name;
    name << std::defaultfloat << std::setprecision(6)
         << parameters.checkpoint_offset + time.current();
    const std::string file =
      parameters.checkpoint_folder + "/" + name.str() + ".solid";
    std::ofstream out(file + ".tmp", std::ios::binary);
    displacement.block_write(out);
    velocity.block_write(out);
    old_stress.block_write(out);
    servo_theta.block_write(out);
    servo_omega.block_write(out);
    servo_cmd.block_write(out);
    out.close();
    AssertThrow(out, ExcMessage("Cannot write the checkpoint " + file));
    std::rename((file + ".tmp").c_str(), file.c_str());
  }



  template <int dim>
  void
  ElastoDynamics<dim>::read_checkpoint(const std::string &file)
  {
    std::ifstream in(file, std::ios::binary);
    AssertThrow(in, ExcMessage("Cannot open the checkpoint " + file));
    const auto read = [&in, &file](Vector<double> &v) {
      const auto n = v.size();
      v.block_read(in);
      AssertThrow(v.size() == n,
                  ExcMessage("The checkpoint " + file +
                             " does not fit this mesh / setup"));
    };
    read(displacement);
    read(velocity);
    read(old_stress);
    read(servo_theta);
    read(servo_omega);
    read(servo_cmd);
    old_displacement = displacement;
    old_velocity     = velocity;
    std::cout << "Restarted from the checkpoint " << file << std::endl;
  }



  template class ElastoDynamics<DIM>;
} // namespace Linear_Elasticity
