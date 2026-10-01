#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <deal.II/base/parameter_handler.h>

/**
 * This class declares all parameters, which can be specified in the parameter
 * file. The subsection abut preCICE configurations is directly interlinked
 * to the Adapter class.
 */
namespace Parameters
{
  using namespace dealii;
  /**
   * @brief Time: Specifies simulation time properties
   */
  struct Time
  {
    double      end_time        = 1;
    double      delta_t         = 0.1;
    int         output_interval = 1;
    std::string output_folder   = "";


    void
    add_output_parameters(ParameterHandler &prm);
  };

  /**
   * @brief The System struct keeps material properties and body force contributions
   */
  struct System
  {
    double               nu     = 0.3;
    double               mu     = 1538462;
    double               lambda = -1;
    double               rho    = 1000;
    Tensor<1, 3, double> body_force;

    void
    add_output_parameters(ParameterHandler &prm);
  };


  /**
   * @brief LinearSolver: Specifies linear solver properties
   */
  struct Solver
  {
    std::string  model              = "linear";
    std::string  type_lin           = "Direct";
    double       tol_lin            = 1e-6;
    double       max_iterations_lin = 1;
    unsigned int max_iterations_NR  = 10;
    double       tol_f              = 1e-9;
    double       tol_u              = 1e-6;

    void
    add_output_parameters(ParameterHandler &prm);
  };



  /**
   * @brief Discretization: Specifies parameters for time integration by a
   *        theta scheme and polynomial degree of the FE system
   */
  struct Discretization
  {
    unsigned int poly_degree = 3;
    // For the linear elastic model (theta-scheme)
    double theta = 0.5;
    // For the nonlinear elastic model (Newmark)
    double beta  = 0.25;
    double gamma = 0.5;

    void
    add_output_parameters(ParameterHandler &prm);
  };


  /**
   * @brief PreciceAdapterConfiguration: Specifies preCICE related information.
   *        A lot of these information need to be consistent with the
   *        precice-config.xml file.
   */
  struct PreciceAdapterConfiguration
  {
    std::string scenario         = "FSI3";
    double      plate_thickness  = 3.0e-3; // Sled foam plate [m]
    std::string config_file      = "precice-config.xml";
    std::string participant_name = "dealiisolver";
    std::string mesh_name        = "dealii-mesh";
    std::string read_data_name   = "Stress";
    std::string write_data_name  = "Displacement";
    double      flap_location    = 0.0;
    bool        data_consistent  = true;

    // Optional actuator-disk propeller loads received from the fluid
    bool        prop_enabled         = false;
    std::string prop_mesh_name       = "Propeller-Mesh";
    std::string prop_force_data_name = "Thrust";
    std::string prop_torque_data_name = "PropTorque";

    // Optional control-surface hinge loads received from the fluid
    bool         hinge_enabled         = false;
    std::string  hinge_mesh_name       = "Control-Mesh";
    std::string  hinge_force_data_name = "HingeForce";
    std::string  hinge_moment_data_name = "HingeMoment";

    // Actuated control-surface hinge: a rotational DOF per surface, driven by a
    // servo (saturated PD position loop). The aero hinge moment is read on the
    // hinge mesh and applied to the DOF instead of the elastic body.
    bool        servo_enabled          = false;
    std::string servo_command_data_name = "ServoCommand";
    std::string servo_angle_data_name   = ""; // written back, optional
    std::vector<double> hinge_axis{0.0, 1.0, 0.0};
    std::vector<double> servo_hinge_locations; // dim-tuples, flat

    // Servo model parameters (defaults are a small-UAV analogue servo).
    double servo_kp         = 2.0;
    double servo_kd         = 0.05;
    double servo_torque_max = 0.30;
    double servo_rate_max   = 6.0;
    double servo_freeplay   = 0.005;
    double servo_coulomb    = 0.010;
    double servo_viscous    = 0.002;
    double servo_inertia    = 2.0e-5;

    void
    add_output_parameters(ParameterHandler &prm);
  };


  struct AllParameters : public Solver,
                         public Discretization,
                         public System,
                         public Time,
                         public PreciceAdapterConfiguration

  {
    AllParameters(const std::string &input_file);
  };
} // namespace Parameters

#endif // PARAMETERS_H
