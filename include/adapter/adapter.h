#ifndef ADAPTER_H
#define ADAPTER_H

#include <deal.II/base/exceptions.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping_q1.h>

#include <adapter/dof_tools_extension.h>
#include <adapter/time_handler.h>
#include <precice/precice.hpp>


namespace Adapter
{
  using namespace dealii;

  /**
   * The Adapter class keeps all functionalities to couple deal.II to other
   * solvers with preCICE i.e. data structures are set up, necessary information
   * is passed to preCICE etc.
   */
  template <int dim, typename VectorType, typename ParameterClass>
  class Adapter
  {
  public:
    /**
     * @brief      Constructor, which sets up the precice Participant
     *
     * @param[in]  parameters Parameter class, which hold the data specified
     *             in the parameters.prm file
     * @param[in]  deal_boundary_interface_id Boundary ID of the triangulation,
     *             which is associated with the coupling interface.
     */
    Adapter(const ParameterClass &parameters,
            const unsigned int    deal_boundary_interface_id);

    /**
     * @brief      Initializes preCICE and passes all relevant data to preCICE
     *
     * @param[in]  dof_handler Initialized dof_handler
     * @param[in]  deal_to_precice Data, which should be given to preCICE and
     *             exchanged with other participants. Wether this data is
     *             required already in the beginning depends on your
     *             individual configuration and preCICE determines it
     *             automatically. In many cases, this data will just represent
     *             your initial condition.
     * @param[in]  relative_read_time Time associated to the coupling data
     *             received from preCICE and stored in \p precice_to_deal.
     *             See also precice::Participant::readData
     * @param[out] precice_to_deal Data, which is received from preCICE/ from
     *             other participants. Wether this data is useful already in
     *             the beginning depends on your individual configuration and
     *             preCICE determines it automatically. In many cases, this
     *             data will just represent the initial condition of other
     *             participants.
     *
     */
    void
    initialize(const DoFHandler<dim> &dof_handler,
               const VectorType &     deal_to_precice);

    /**
     * @brief Fetches the read data from preCICE for the next time-step
     *
     * @param[in]  relative_read_time Time associated to the coupling data
     *             received from preCICE and stored in \p precice_to_deal.
     *             See also precice::Participant::readData
     * @param[out] precice_to_deal Same data as in @p initialize_precice() i.e.
     *             data, which is received from preCICE/other participants
     *             after each time step and exchanged with other participants.
     */
    void
    read_data(double relative_read_time, VectorType &precice_to_deal);

    /**
     * @brief      Advances preCICE after every timestep, converts data formats
     *             between preCICE and dealii
     *
     * @param[in]  deal_to_precice Same data as in @p initialize_precice() i.e.
     *             data, which should be given to preCICE after each time step
     *             and exchanged with other participants.
     * @param[in]  computed_timestep_length Length of the timestep used by
     *             the solver.
     */
    void
    advance(const VectorType &deal_to_precice,
            const double      computed_timestep_length);

    /**
     * @brief      Saves current state of time dependent variables in case of an
     *             implicit coupling
     *
     * @param[in]  state_variables Vector containing all variables to store as
     *             reference
     *
     * @note       This function only makes sense, if it is used with
     *             @p reload_old_state_if_required. Therefore, the order, in which the
     *             variables are passed into the vector must be the same for
     *             both functions.
     * @note       The absolute time has no impact on the computation, but on the output.
     *             Therefore, we call here in the @p Time class a method to store the
     *             current time and reload it later. This is necessary, in
     *             case your solver is subcycling.
     */
    void
    save_current_state_if_required(
      const std::vector<VectorType *> &state_variables,
      Time &                           time_class);

    /**
     * @brief      Reloads the previously stored variables in case of an implicit
     *             coupling. The current implementation supports subcycling,
     *             i.e. previously refers o the last time
     *             @p save_current_state_if_required() has been called.
     *
     * @param[out] state_variables Vector containing all variables to reload
     *             as reference
     *
     * @note       This function only makes sense, if the state variables have been
     *             stored by calling @p save_current_state_if_required. Therefore,
     *             the order, in which the variables are passed into the
     *             vector must be the same for both functions.
     */
    void
    reload_old_state_if_required(std::vector<VectorType *> &state_variables,
                                 Time &                     time_class);

    /**
     * @brief      Enables the optional coupling of actuator-disk propeller
     *             loads (one thrust force and one reaction torque per hub),
     *             which are received on a separate preCICE mesh provided by
     *             the fluid participant.
     *
     * @param[in]  prop_mesh_name Name of the received propeller hub mesh
     *             in the precice-config file.
     * @param[in]  prop_force_name Name of the (vector) data with the thrust
     *             per hub on the propeller mesh.
     * @param[in]  prop_torque_name Name of the (vector) data with the torque
     *             per hub on the propeller mesh.
     */
    void
    configure_propeller(const std::string &prop_mesh_name,
                        const std::string &prop_force_name,
                        const std::string &prop_torque_name);

    /**
     * @brief Define the access region for the received propeller mesh.
     *        Must be called before @p initialize() (as preCICE requires the
     *        geometry access to be defined before the initialization).
     */
    void
    set_prop_mesh_access_region();

    /**
     * @brief Fetch the received propeller mesh vertices (hub coordinates)
     *        from preCICE. Must be called after @p initialize().
     */
    void
    initialize_prop_mesh();

    /**
     * @brief Read the propeller force and torque data (per hub) for the given
     *        relative read time.
     *
     * @param[in]  relative_read_time Time associated to the coupling data
     *             received from preCICE.
     * @param[out] prop_force  dim * nProp vector with the thrust per hub.
     * @param[out] prop_torque dim * nProp vector with the torque per hub.
     */
    void
    read_prop_data(double               relative_read_time,
                   std::vector<double> &prop_force,
                   std::vector<double> &prop_torque);

    /**
     * @brief      Enables the optional coupling of control-surface hinge
     *             loads (one force and one moment about the hinge per
     *             surface), received on a separate preCICE mesh provided by
     *             the fluid participant.
     *
     * @param[in]  hinge_mesh_name Name of the received hinge mesh in the
     *             precice-config file.
     * @param[in]  hinge_force_name Name of the (vector) data with the hinge
     *             force per surface.
     * @param[in]  hinge_moment_name Name of the (vector) data with the hinge
     *             moment per surface.
     */
    void
    configure_hinges(const std::string &hinge_mesh_name,
                     const std::string &hinge_force_name,
                     const std::string &hinge_moment_name);

    /**
     * @brief Define the access region for the received hinge mesh.
     *        Must be called before @p initialize().
     */
    void
    set_hinge_mesh_access_region();

    /**
     * @brief Fetch the received hinge mesh vertices. Must be called after
     *        @p initialize().
     */
    void
    initialize_hinge_mesh();

    /**
     * @brief Read the hinge force and moment data (per surface) for the given
     *        relative read time.
     */
    void
    read_hinge_data(double               relative_read_time,
                    std::vector<double> &hinge_force,
                    std::vector<double> &hinge_moment);

    /**
     * @brief Enable reading of the servo command angle per control surface.
     *        The data is received on the hinge mesh (one scalar per surface).
     */
    void
    configure_servo_command(const std::string &command_name);

    /**
     * @brief Read the servo command angles (one per surface) for the given
     *        relative read time.
     */
    void
    read_servo_command(double               relative_read_time,
                       std::vector<double> &command);

    /**
     * @brief Enable writing of the actual servo angle per control surface on
     *        the hinge mesh (one scalar per surface), so the fluid can move
     *        the surface by the angle the servo reached, not the command.
     */
    void
    configure_servo_angle(const std::string &angle_name);

    /**
     * @brief Write the servo angles (one per surface, in hinge mesh vertex
     *        order). Call before advance().
     */
    void
    write_servo_angle(const std::vector<double> &angle);

    /**
     * @brief Number of hinge vertices received from preCICE.
     */
    unsigned int
    get_n_hinge_vertices() const;

    /**
     * @brief Coordinates of the hinge vertices as received from preCICE
     *        (dim * n_hinges values, interleaved).
     */
    const std::vector<double> &
    get_hinge_vertices_coords() const;

    /**
     * @brief Number of propeller hubs received from preCICE.
     */
    unsigned int
    get_n_prop_vertices() const;

    /**
     * @brief Coordinates of the propeller hubs as received from preCICE
     *        (dim*n_hubs values, interleaved).
     */
    const std::vector<double> &
    get_prop_vertices_coords() const;

    /**
     * @brief public precice Participant
     */

    precice::Participant precice;

    // Boundary ID of the deal.II mesh, associated with the coupling
    // interface. The variable is public and should be used during grid
    // generation, but is also involved during system assembly. The only thing,
    // one needs to make sure is, that this ID is not given to another part of
    // the boundary e.g. clamped one.
    const unsigned int deal_boundary_interface_id;

  private:
    // preCICE related initializations
    // These variables are specified and read from the parameter file
    const std::string mesh_name;
    const std::string read_data_name;
    const std::string write_data_name;

    // To be adjusted for MPI parallelized codes
    static constexpr unsigned int this_mpi_process = 0;
    static constexpr unsigned int n_mpi_processes  = 1;

    // These IDs are given by preCICE during initialization
    int n_interface_nodes;

    // Dof IndexSets of the global deal.II vectors, containing relevant
    // coupling dof indices
    IndexSet coupling_dofs_x_comp;
    IndexSet coupling_dofs_y_comp;
    IndexSet coupling_dofs_z_comp;

    // Data containers which are passed to preCICE in an appropriate preCICE
    // specific format
    std::vector<int>    interface_nodes_ids;
    std::vector<double> read_data_buffer;
    std::vector<double> write_data_buffer;

    // Container to store time dependent data in case of an implicit coupling
    std::vector<VectorType> old_state_data;
    double                  old_time_value;

    // Optional coupling of actuator-disk propeller loads
    bool                prop_enabled    = false;
    std::string         prop_mesh_name;
    std::string         prop_force_name;
    std::string         prop_torque_name;
    std::vector<int>    prop_vertex_ids;
    std::vector<double> prop_vertex_coords;

    // Optional coupling of control-surface hinge loads
    bool                hinge_enabled     = false;
    std::string         hinge_mesh_name;
    std::string         hinge_force_name;
    std::string         hinge_moment_name;
    std::vector<int>    hinge_vertex_ids;
    std::vector<double> hinge_vertex_coords;

    // Optional servo command read on the hinge mesh
    bool        servo_command_enabled = false;
    std::string servo_command_name;

    // Optional servo angle written on the hinge mesh
    bool        servo_angle_enabled = false;
    std::string servo_angle_name;

    /**
     * @brief format_deal_to_precice Formats a global deal.II vector of type
     *        VectorType to a std::vector for preCICE. This functions is only
     *        used internally in the class and should not be called in the
     *        solver.
     *
     * @param[in] deal_to_precice Global deal.II vector of VectorType. The
     *            result (preCICE specific vector) is stored in the class in
     *            the variable 'write_data_buffer'.
     *
     * @note  The order, in which preCICE obtains data from the solver, needs
     *        to be consistent with the order of the initially passed vertices
     *        coordinates.
     */
    void
    format_deal_to_precice(const VectorType &deal_to_precice);

    /**
     * @brief format_precice_to_deal Takes the std::vector obtained by preCICE
     *        in 'read_data_buffer' and inserts the values to the right position
     *        in the global deal.II vector of size n_global_dofs. This is the
     *        opposite functionality as @p foramt_precice_to_deal(). This
     *        functions is only used internally in the class and should not
     *        be called in the solver.
     *
     * @param[out] precice_to_deal Global deal.II vector of VectorType and
     *             size n_global_dofs.
     *
     * @note  The order, in which preCICE obtains data from the solver, needs
     *        to be consistent with the order of the initially passed vertices
     *        coordinates.
     */
    void
    format_precice_to_deal(VectorType &precice_to_deal) const;
  };



  template <int dim, typename VectorType, typename ParameterClass>
  Adapter<dim, VectorType, ParameterClass>::Adapter(
    const ParameterClass &parameters,
    const unsigned int    deal_boundary_interface_id)
    : precice(parameters.participant_name,
              parameters.config_file,
              this_mpi_process,
              n_mpi_processes)
    , deal_boundary_interface_id(deal_boundary_interface_id)
    , mesh_name(parameters.mesh_name)
    , read_data_name(parameters.read_data_name)
    , write_data_name(parameters.write_data_name)
  {}



  template <int dim, typename VectorType, typename ParameterClass>
  void
  Adapter<dim, VectorType, ParameterClass>::initialize(
    const DoFHandler<dim> &dof_handler,
    const VectorType &     deal_to_precice)
  {
    AssertThrow(
      dim == precice.getMeshDimensions(mesh_name),
      ExcMessage("The dimension of your solver needs to be consistent with the "
                 "dimension specified in your precice-config file. In case you "
                 "run one of the tutorials, the dimension can be specified via "
                 "cmake -D DIM=dim ."));

    AssertThrow(dim > 1, ExcNotImplemented());

    // get the number of interface nodes from deal.II
    // Therefore, we extract one component of the vector valued dofs and store
    // them in an IndexSet
    std::set<types::boundary_id> couplingBoundary{deal_boundary_interface_id};
    // Return by value for newer deal.II versions
#if DEAL_II_VERSION_GTE(9, 3, 0)
    auto get_component_dofs = [&](const int component) {
      const FEValuesExtractors::Scalar component_dofs(component);
      return DoFTools::extract_boundary_dofs(
        dof_handler,
        dof_handler.get_fe().component_mask(component_dofs),
        couplingBoundary);
    };
    coupling_dofs_x_comp = get_component_dofs(0);
    coupling_dofs_y_comp = get_component_dofs(1);
    if (dim == 3)
      coupling_dofs_z_comp = get_component_dofs(2);
#else
    // Return by argument for older deal.II versions
    auto get_component_dofs = [&](const int component, auto &dof_index_set) {
      const FEValuesExtractors::Scalar component_dofs(component);
      DoFTools::extract_boundary_dofs(dof_handler,
                                      dof_handler.get_fe().component_mask(
                                        component_dofs),
                                      dof_index_set,
                                      couplingBoundary);
    };
    get_component_dofs(0, coupling_dofs_x_comp);
    get_component_dofs(1, coupling_dofs_y_comp);
    if (dim == 3)
      get_component_dofs(2, coupling_dofs_z_comp);
#endif
    n_interface_nodes = coupling_dofs_x_comp.n_elements();

    std::cout << "\t Number of coupling nodes:     " << n_interface_nodes
              << std::endl;

    // Set up a vector to pass the node positions to preCICE. Each node is
    // specified once. One needs to specify in the precice-config.xml, whether
    // the data is vector valued or not.
    std::vector<double> interface_nodes_positions(dim * n_interface_nodes);

    // Set up the appropriate size of the data container needed for data
    // exchange. Here, we deal with a vector valued problem for read and write
    // data namely displacement and forces. Therefore, we need dim entries per
    // vertex
    write_data_buffer.resize(dim * n_interface_nodes);
    read_data_buffer.resize(dim * n_interface_nodes);
    interface_nodes_ids.resize(n_interface_nodes);

    // get the coordinates of the interface nodes from deal.ii
    std::map<types::global_dof_index, Point<dim>> support_points;

    // We use here a simple Q1 mapping. In case one has more complex
    // geomtries, you might want to change this to a higher order mapping.
    // We only need to map the first component for a dim dimensional problem
    const FEValuesExtractors::Scalar dofs_x_direction(0);
    DoFTools::map_boundary_dofs_to_support_points(
      StaticMappingQ1<dim>::mapping,
      dof_handler,
      support_points,
      dof_handler.get_fe().component_mask(dofs_x_direction),
      deal_boundary_interface_id);

    // support_points contains now the coordinates of all dofs
    // in the next step, the relevant coordinates are extracted using the
    // IndexSet with the extracted coupling_dofs.

    // preCICE expects all data in the format [x0, y0, z0, x1, y1 ...]
    int node_position_iterator = 0;
    for (auto element : coupling_dofs_x_comp)
      {
        for (int i = 0; i < dim; ++i)
          interface_nodes_positions[node_position_iterator * dim + i] =
            support_points[element][i];

        ++node_position_iterator;
      }

    // pass node coordinates to precice
    precice.setMeshVertices(mesh_name,
                            interface_nodes_positions,
                            interface_nodes_ids);

    // write initial writeData to preCICE if required
    if (precice.requiresInitialData())
      {
        // store initial write_data for precice in write_data_buffer
        format_deal_to_precice(deal_to_precice);

        precice.writeData(mesh_name,
                          write_data_name,
                          interface_nodes_ids,
                          write_data_buffer);
      }

    // Initialize preCICE internally
    precice.initialize();
  }



  template <int dim, typename VectorType, typename ParameterClass>
  void
  Adapter<dim, VectorType, ParameterClass>::read_data(
    double      relative_read_time,
    VectorType &precice_to_deal)
  {
    // Here, we obtain data from another participant. Again, we insert the
    // data in our global vector by calling format_precice_to_deal
    precice.readData(mesh_name,
                     read_data_name,
                     interface_nodes_ids,
                     relative_read_time,
                     read_data_buffer);

    format_precice_to_deal(precice_to_deal);
  }



  template <int dim, typename VectorType, typename ParameterClass>
  void
  Adapter<dim, VectorType, ParameterClass>::advance(
    const VectorType &deal_to_precice,
    const double      computed_timestep_length)
  {
    // This is essentially the same as during initialization
    // We have already all IDs and just need to convert our obtained data to
    // the preCICE compatible 'write_data_buffer' vector, which is done in the
    // format_deal_to_precice function.
    format_deal_to_precice(deal_to_precice);

    precice.writeData(mesh_name,
                      write_data_name,
                      interface_nodes_ids,
                      write_data_buffer);

    // Here, we need to specify the computed time step length and pass it to
    // preCICE
    precice.advance(computed_timestep_length);
  }



  template <int dim, typename VectorType, typename ParameterClass>
  void
  Adapter<dim, VectorType, ParameterClass>::format_deal_to_precice(
    const VectorType &deal_to_precice)
  {
    // Assumption: x index is in the same position as y index in each IndexSet
    // In general, higher order support points in the element are first
    // ordered in the x component. An IndexSet for the first component might
    // look like this: [1] [3456] [11] for a 7th order 1d interface/2d cell.
    // Therefore, an index for the respective x component dof is not always
    // followed by an index on the same position for the y component

    auto x_comp = coupling_dofs_x_comp.begin();
    auto y_comp = coupling_dofs_y_comp.begin();
    auto z_comp = coupling_dofs_z_comp.begin();

    for (int i = 0; i < n_interface_nodes; ++i)
      {
        write_data_buffer[dim * i]       = deal_to_precice[*x_comp];
        write_data_buffer[(dim * i) + 1] = deal_to_precice[*y_comp];
        ++x_comp;
        ++y_comp;
        if (dim == 3)
          {
            write_data_buffer[(dim * i) + 2] = deal_to_precice[*z_comp];
            ++z_comp;
          }
      }
  }



  template <int dim, typename VectorType, typename ParameterClass>
  void
  Adapter<dim, VectorType, ParameterClass>::format_precice_to_deal(
    VectorType &precice_to_deal) const
  {
    // This is the opposite direction as above. See comment there.
    auto x_comp = coupling_dofs_x_comp.begin();
    auto y_comp = coupling_dofs_y_comp.begin();
    auto z_comp = coupling_dofs_z_comp.begin();

    for (int i = 0; i < n_interface_nodes; ++i)
      {
        precice_to_deal[*x_comp] = read_data_buffer[dim * i];
        precice_to_deal[*y_comp] = read_data_buffer[(dim * i) + 1];
        ++x_comp;
        ++y_comp;
        if (dim == 3)
          {
            precice_to_deal[*z_comp] = read_data_buffer[(dim * i) + 2];
            ++z_comp;
          }
      }
  }



  template <int dim, typename VectorType, typename ParameterClass>
  void
  Adapter<dim, VectorType, ParameterClass>::save_current_state_if_required(
    const std::vector<VectorType *> &state_variables,
    Time &                           time_class)
  {
    // First, we let preCICE check, whether we need to store the variables.
    // Then, the data is stored in the class
    if (precice.requiresWritingCheckpoint())
      {
        old_state_data.resize(state_variables.size());

        for (uint i = 0; i < state_variables.size(); ++i)
          old_state_data[i] = *(state_variables[i]);

        old_time_value = time_class.current();
      }
  }



  template <int dim, typename VectorType, typename ParameterClass>
  void
  Adapter<dim, VectorType, ParameterClass>::reload_old_state_if_required(
    std::vector<VectorType *> &state_variables,
    Time &                     time_class)
  {
    // In case we need to reload a state, we just take the internally stored
    // data vectors and write then in to the input data
    if (precice.requiresReadingCheckpoint())
      {
        Assert(state_variables.size() == old_state_data.size(),
               ExcMessage(
                 "state_variables are not the same as previously saved."));

        for (uint i = 0; i < state_variables.size(); ++i)
          *(state_variables[i]) = old_state_data[i];

        // Here, we expect the time class to offer an option to specify a
        // given time value.
        time_class.set_absolute_time(old_time_value);
      }
  }



  // ----------------------------------------------------------------------- //
  // Optional coupling of actuator-disk propeller loads
  // ----------------------------------------------------------------------- //

  template <int dim, typename VectorType, typename ParameterClass>
  void
  Adapter<dim, VectorType, ParameterClass>::configure_propeller(
    const std::string &prop_mesh_name,
    const std::string &prop_force_name,
    const std::string &prop_torque_name)
  {
    prop_enabled         = true;
    this->prop_mesh_name = prop_mesh_name;
    this->prop_force_name  = prop_force_name;
    this->prop_torque_name = prop_torque_name;
  }



  template <int dim, typename VectorType, typename ParameterClass>
  void
  Adapter<dim, VectorType, ParameterClass>::set_prop_mesh_access_region()
  {
    if (!prop_enabled)
      return;

    // Define a region of interest covering the whole domain. The hub
    // vertices of the received propeller mesh are expected there.
    std::vector<double> bounding_box;
    const int           prop_dim = precice.getMeshDimensions(prop_mesh_name);
    if (prop_dim == 2)
      bounding_box = {-1.0e9, 1.0e9, -1.0e9, 1.0e9};
    else
      bounding_box = {-1.0e9, 1.0e9, -1.0e9, 1.0e9, -1.0e9, 1.0e9};

    precice.setMeshAccessRegion(prop_mesh_name, bounding_box);
  }



  template <int dim, typename VectorType, typename ParameterClass>
  void
  Adapter<dim, VectorType, ParameterClass>::initialize_prop_mesh()
  {
    if (!prop_enabled)
      return;

    const int prop_dim = precice.getMeshDimensions(prop_mesh_name);
    const int n_props  = precice.getMeshVertexSize(prop_mesh_name);

    AssertThrow(prop_dim == dim,
                ExcMessage("Propeller mesh dimension mismatch."));

    prop_vertex_ids.resize(n_props);
    prop_vertex_coords.resize(n_props * prop_dim);

    precice.getMeshVertexIDsAndCoordinates(
      prop_mesh_name, prop_vertex_ids, prop_vertex_coords);

    std::cout << "\t Number of propeller hubs: " << n_props << std::endl;
  }



  template <int dim, typename VectorType, typename ParameterClass>
  void
  Adapter<dim, VectorType, ParameterClass>::read_prop_data(
    double               relative_read_time,
    std::vector<double> &prop_force,
    std::vector<double> &prop_torque)
  {
    if (!prop_enabled)
      return;

    const int prop_dim = precice.getMeshDimensions(prop_mesh_name);
    const int n_prop   = prop_vertex_ids.size();

    prop_force.resize(n_prop * prop_dim);
    prop_torque.resize(n_prop * prop_dim);

    precice.readData(prop_mesh_name,
                     prop_force_name,
                     prop_vertex_ids,
                     relative_read_time,
                     prop_force);

    precice.readData(prop_mesh_name,
                     prop_torque_name,
                     prop_vertex_ids,
                     relative_read_time,
                     prop_torque);
  }



  template <int dim, typename VectorType, typename ParameterClass>
  unsigned int
  Adapter<dim, VectorType, ParameterClass>::get_n_prop_vertices() const
  {
    return prop_vertex_ids.size();
  }



  template <int dim, typename VectorType, typename ParameterClass>
  const std::vector<double> &
  Adapter<dim, VectorType, ParameterClass>::get_prop_vertices_coords() const
  {
    return prop_vertex_coords;
  }



  // ----------------------------------------------------------------------- //
  // Optional coupling of control-surface hinge loads
  // ----------------------------------------------------------------------- //

  template <int dim, typename VectorType, typename ParameterClass>
  void
  Adapter<dim, VectorType, ParameterClass>::configure_hinges(
    const std::string &hinge_mesh_name,
    const std::string &hinge_force_name,
    const std::string &hinge_moment_name)
  {
    hinge_enabled          = true;
    this->hinge_mesh_name  = hinge_mesh_name;
    this->hinge_force_name = hinge_force_name;
    this->hinge_moment_name = hinge_moment_name;
  }



  template <int dim, typename VectorType, typename ParameterClass>
  void
  Adapter<dim, VectorType, ParameterClass>::set_hinge_mesh_access_region()
  {
    if (!hinge_enabled)
      return;

    std::vector<double> bounding_box;
    const int           hinge_dim = precice.getMeshDimensions(hinge_mesh_name);
    if (hinge_dim == 2)
      bounding_box = {-1.0e9, 1.0e9, -1.0e9, 1.0e9};
    else
      bounding_box = {-1.0e9, 1.0e9, -1.0e9, 1.0e9, -1.0e9, 1.0e9};

    precice.setMeshAccessRegion(hinge_mesh_name, bounding_box);
  }



  template <int dim, typename VectorType, typename ParameterClass>
  void
  Adapter<dim, VectorType, ParameterClass>::initialize_hinge_mesh()
  {
    if (!hinge_enabled)
      return;

    const int hinge_dim = precice.getMeshDimensions(hinge_mesh_name);
    const int n_hinges  = precice.getMeshVertexSize(hinge_mesh_name);

    AssertThrow(hinge_dim == dim,
                ExcMessage("Hinge mesh dimension mismatch."));

    hinge_vertex_ids.resize(n_hinges);
    hinge_vertex_coords.resize(n_hinges * hinge_dim);

    precice.getMeshVertexIDsAndCoordinates(
      hinge_mesh_name, hinge_vertex_ids, hinge_vertex_coords);

    std::cout << "\t Number of control-surface hinges: " << n_hinges
              << std::endl;
  }



  template <int dim, typename VectorType, typename ParameterClass>
  void
  Adapter<dim, VectorType, ParameterClass>::read_hinge_data(
    double               relative_read_time,
    std::vector<double> &hinge_force,
    std::vector<double> &hinge_moment)
  {
    if (!hinge_enabled)
      return;

    const int hinge_dim = precice.getMeshDimensions(hinge_mesh_name);
    const int n_hinges  = hinge_vertex_ids.size();

    hinge_force.resize(n_hinges * hinge_dim);
    hinge_moment.resize(n_hinges * hinge_dim);

    precice.readData(hinge_mesh_name,
                     hinge_force_name,
                     hinge_vertex_ids,
                     relative_read_time,
                     hinge_force);

    precice.readData(hinge_mesh_name,
                     hinge_moment_name,
                     hinge_vertex_ids,
                     relative_read_time,
                     hinge_moment);
  }



  template <int dim, typename VectorType, typename ParameterClass>
  void
  Adapter<dim, VectorType, ParameterClass>::configure_servo_command(
    const std::string &command_name)
  {
    servo_command_enabled = true;
    servo_command_name    = command_name;
  }



  template <int dim, typename VectorType, typename ParameterClass>
  void
  Adapter<dim, VectorType, ParameterClass>::read_servo_command(
    double               relative_read_time,
    std::vector<double> &command)
  {
    if (!servo_command_enabled)
      return;
    const int n_hinges = hinge_vertex_ids.size();
    if (n_hinges == 0)
      return;
    command.resize(n_hinges);
    precice.readData(hinge_mesh_name,
                     servo_command_name,
                     hinge_vertex_ids,
                     relative_read_time,
                     command);
  }



  template <int dim, typename VectorType, typename ParameterClass>
  void
  Adapter<dim, VectorType, ParameterClass>::configure_servo_angle(
    const std::string &angle_name)
  {
    servo_angle_enabled = !angle_name.empty();
    servo_angle_name    = angle_name;
  }



  template <int dim, typename VectorType, typename ParameterClass>
  void
  Adapter<dim, VectorType, ParameterClass>::write_servo_angle(
    const std::vector<double> &angle)
  {
    if (!servo_angle_enabled || hinge_vertex_ids.empty())
      return;
    AssertThrow(angle.size() == hinge_vertex_ids.size(),
                ExcMessage("One servo angle per hinge vertex is required."));
    precice.writeData(hinge_mesh_name,
                      servo_angle_name,
                      hinge_vertex_ids,
                      angle);
  }



  template <int dim, typename VectorType, typename ParameterClass>
  unsigned int
  Adapter<dim, VectorType, ParameterClass>::get_n_hinge_vertices() const
  {
    return hinge_vertex_ids.size();
  }



  template <int dim, typename VectorType, typename ParameterClass>
  const std::vector<double> &
  Adapter<dim, VectorType, ParameterClass>::get_hinge_vertices_coords() const
  {
    return hinge_vertex_coords;
  }
} // namespace Adapter

#endif // ADAPTER_H
