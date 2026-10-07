#include <adapter/parameters.h>

namespace Parameters
{
  void
  Time::add_output_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Time");
    {
      prm.add_parameter("End time", end_time, "End time", Patterns::Double());

      prm.add_parameter("Time step size",
                        delta_t,
                        "Time step size",
                        Patterns::Double());

      prm.add_parameter("Output interval",
                        output_interval,
                        "Write results every x timesteps",
                        Patterns::Integer(0));
      prm.add_parameter("Output folder",
                        output_folder,
                        "Output folder",
                        Patterns::Anything());
      prm.add_parameter("Checkpoint interval",
                        checkpoint_interval,
                        "Write a restart checkpoint every x seconds (0: none)",
                        Patterns::Double(0));
      prm.add_parameter("Checkpoint folder",
                        checkpoint_folder,
                        "Folder of the restart checkpoints",
                        Patterns::Anything());
      prm.add_parameter("Checkpoint time offset",
                        checkpoint_offset,
                        "Added to the time in checkpoint names (the start "
                        "time of a restarted run)",
                        Patterns::Double());
      prm.add_parameter("Restart file",
                        restart_file,
                        "Checkpoint to start from (empty: from rest)",
                        Patterns::Anything());
    }
    prm.leave_subsection();
  }


  void
  System::add_output_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("System properties");
    {
      prm.add_parameter("Shear modulus",
                        mu,
                        "Shear modulus",
                        Patterns::Double());

      prm.add_parameter("Poisson's ratio",
                        nu,
                        "Poisson's ratio",
                        Patterns::Double(-1.0, 0.5));

      prm.add_parameter("rho", rho, "density", Patterns::Double(0.0));

      prm.add_parameter("body forces",
                        body_force,
                        "body forces x,y,z",
                        Patterns::List(Patterns::Double()));
    }
    prm.leave_subsection();
  }


  void
  Solver::add_output_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Solver");
    {
      prm.add_parameter("Model",
                        model,
                        "Structural model to be used: linear or neo-Hookean",
                        Patterns::Selection("linear|neo-Hookean"));

      prm.add_parameter("Solver type",
                        type_lin,
                        "Linear solver: CG or Direct",
                        Patterns::Selection("CG|Direct"));

      prm.add_parameter(
        "Residual",
        tol_lin,
        "CG solver residual (multiplied by residual norm, ignored if Model == linear)",
        Patterns::Double(0.0));

      prm.add_parameter(
        "Max iteration multiplier",
        max_iterations_lin,
        "Max CG solver iterations (multiples of the system matrix size)",
        Patterns::Double(0.0));

      prm.add_parameter(
        "Max iterations Newton-Raphson",
        max_iterations_NR,
        "Number of Newton-Raphson iterations allowed (ignored if Model == linear)",
        Patterns::Integer(0));

      prm.add_parameter(
        "Tolerance force",
        tol_f,
        "Force residual tolerance for non-linear iteration (ignored if Model == linear)",
        Patterns::Double(0.0));

      prm.add_parameter(
        "Tolerance displacement",
        tol_u,
        "Displacement error tolerance for non-linear iteration (ignored if Model == linear)",
        Patterns::Double(0.0));
    }
    prm.leave_subsection();
  }


  void
  Discretization::add_output_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Discretization");
    {
      prm.add_parameter("Polynomial degree",
                        poly_degree,
                        "Polynomial degree of the FE system",
                        Patterns::Integer(0));

      prm.add_parameter("theta",
                        theta,
                        "Time integration scheme",
                        Patterns::Double(0, 1));

      prm.add_parameter("beta", beta, "Newmark beta", Patterns::Double(0, 0.5));

      prm.add_parameter("gamma",
                        gamma,
                        "Newmark gamma",
                        Patterns::Double(0, 1));
    }
    prm.leave_subsection();
  }


  void
  PreciceAdapterConfiguration::add_output_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("precice configuration");
    {
      prm.add_parameter("Scenario",
                        scenario,
                        "Cases: FSI3, PF for perpendicular flap, or Sled for the "
                        "flying-sled foam airframe plate",
                        Patterns::Selection("FSI3|PF|Sled"));

      prm.add_parameter("Plate thickness",
                        plate_thickness,
                        "Thickness of the Sled foam plate [m]",
                        Patterns::Double(0));

      prm.add_parameter("precice config-file",
                        config_file,
                        "Name of the precice configuration file",
                        Patterns::Anything());

      prm.add_parameter(
        "Participant name",
        participant_name,
        "Name of the participant in the precice-config.xml file",
        Patterns::Anything());

      prm.add_parameter(
        "Mesh name",
        mesh_name,
        "Name of the coupling mesh in the precice-config.xml file",
        Patterns::Anything());

      prm.add_parameter("Read data name",
                        read_data_name,
                        "Name of the read data in the precice-config.xml file",
                        Patterns::Anything());

      prm.add_parameter("Write data name",
                        write_data_name,
                        "Name of the write data in the precice-config.xml file",
                        Patterns::Anything());

      prm.add_parameter(
        "Write velocity data name",
        write_velocity_data_name,
        "Name of the interface velocity write data in the precice-config.xml "
        "file (empty: not written)",
        Patterns::Anything());

      prm.add_parameter("Flap location",
                        flap_location,
                        "PF x-location",
                        Patterns::Double(-3, 3));

      // Optional propeller loads (received from the fluid)
      prm.add_parameter("Enable propeller loads",
                        prop_enabled,
                        "Read actuator-disk propeller loads via preCICE",
                        Patterns::Bool());

      prm.add_parameter("Propeller mesh name",
                        prop_mesh_name,
                        "Name of the received propeller hub mesh",
                        Patterns::Anything());

      prm.add_parameter("Propeller force data name",
                        prop_force_data_name,
                        "Name of the thrust data on the propeller mesh",
                        Patterns::Anything());

      prm.add_parameter("Propeller torque data name",
                        prop_torque_data_name,
                        "Name of the torque data on the propeller mesh",
                        Patterns::Anything());

      // Optional control-surface hinge loads
      prm.add_parameter("Enable hinge loads",
                        hinge_enabled,
                        "Read control-surface hinge loads via preCICE",
                        Patterns::Bool());

      prm.add_parameter("Hinge mesh name",
                        hinge_mesh_name,
                        "Name of the received hinge mesh",
                        Patterns::Anything());

      prm.add_parameter("Hinge force data name",
                        hinge_force_data_name,
                        "Name of the hinge force data",
                        Patterns::Anything());

      prm.add_parameter("Hinge moment data name",
                        hinge_moment_data_name,
                        "Name of the hinge moment data",
                        Patterns::Anything());

      // Actuated control-surface hinge (rotational DOF driven by a servo). The
      // servo model is the one shared with the mock fluid (see actuators.hpp):
      // a saturated PD position loop with a slew rate, freeplay, friction and
      // the surface's own rotary inertia.
      prm.add_parameter("Enable servo hinge",
                        servo_enabled,
                        "Drive the control surfaces with a rotational servo DOF",
                        Patterns::Bool());

      prm.add_parameter("Servo command data name",
                        servo_command_data_name,
                        "Name of the received servo command (angle)",
                        Patterns::Anything());

      prm.add_parameter("Servo angle data name",
                        servo_angle_data_name,
                        "Name of the written servo angle (the actual surface "
                        "angle, one scalar per hinge on the hinge mesh); "
                        "empty to not write it",
                        Patterns::Anything());

      prm.add_parameter("Hinge axis",
                        hinge_axis,
                        "Hinge line direction in the solid frame; the surface "
                        "rotates about its own axis (3 components; the "
                        "servo hinge is 3D only)",
                        Patterns::List(Patterns::Double(), 3, 3));

      prm.add_parameter("Servo hinge locations",
                        servo_hinge_locations,
                        "One hinge point per control surface, in the solid "
                        "frame, as a flat list of dim-tuples",
                        Patterns::List(Patterns::Double()));

      prm.add_parameter("Servo stiffness",
                        servo_kp,
                        "Position-loop stiffness [N m / rad]",
                        Patterns::Double(0));

      prm.add_parameter("Servo damping",
                        servo_kd,
                        "Position-loop damping [N m s / rad]",
                        Patterns::Double(0));

      prm.add_parameter("Servo stall torque",
                        servo_torque_max,
                        "Maximum actuator torque [N m]",
                        Patterns::Double(0));

      prm.add_parameter("Servo rate",
                        servo_rate_max,
                        "Maximum commanded slew rate [rad/s]",
                        Patterns::Double(0));

      prm.add_parameter("Servo freeplay",
                        servo_freeplay,
                        "Linkage backlash half-width [rad]",
                        Patterns::Double(0));

      prm.add_parameter("Servo friction",
                        servo_coulomb,
                        "Coulomb friction torque [N m]",
                        Patterns::Double(0));

      prm.add_parameter("Servo viscous friction",
                        servo_viscous,
                        "Viscous friction [N m s / rad]",
                        Patterns::Double(0));

      prm.add_parameter("Servo inertia",
                        servo_inertia,
                        "Reflected servo/flap rotary inertia [kg m^2]",
                        Patterns::Double(0));
    }
    prm.leave_subsection();
  }


  AllParameters::AllParameters(const std::string &input_file)
  {
    ParameterHandler prm;

    Solver::add_output_parameters(prm);
    Discretization::add_output_parameters(prm);
    System::add_output_parameters(prm);
    Time::add_output_parameters(prm);
    PreciceAdapterConfiguration::add_output_parameters(prm);

    prm.parse_input(input_file);

    lambda = 2 * mu * nu / (1 - 2 * nu);

    // Look at the specific type of read data
    if ((read_data_name.find("Stress") == 0))
      data_consistent = true;
    else if ((read_data_name.find("Force") == 0))
      data_consistent = false;
    else
      AssertThrow(
        false,
        ExcMessage(
          "Unknown read data type. Please use 'Force' or 'Stress' in the read data naming."));

    // Optional, if we want to print all parameters in the beginning of the
    // simulation
    //      prm.print_parameters(std::cout,ParameterHandler::Text);
  }
} // namespace Parameters
