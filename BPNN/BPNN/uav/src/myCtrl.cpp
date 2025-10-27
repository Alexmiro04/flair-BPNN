#include "myCtrl.h"
#include <Matrix.h>
#include <Vector3D.h>
#include <TabWidget.h>
#include <CheckBox.h>
#include <Quaternion.h>
#include <Layout.h>
#include <LayoutPosition.h>
#include <GroupBox.h>
#include <DoubleSpinBox.h>
#include <DataPlot1D.h>
#include <cmath>
#include <algorithm>
#include <Euler.h>
#include <iostream>
#include <Label.h>
#include <Vector3DSpinBox.h>
#include <Pid.h>

using std::string;
using namespace flair::core;
using namespace flair::gui;
using namespace flair::filter;

MyController::MyController(const LayoutPosition *position, const string &name) : ControlLaw(position->getLayout(),name,4)
{
    first_update = true;
    has_prev_velocity = false;
    rng.seed(0);
    network_ready = false;
    hidden_neurons = 0;
    last_weight_std = 0.0f;
    b2 = 0.0f;
    theta_hat = 0.0f;
    mass_hat = 0.0f;
    prev_vel_z = 0.0f;
    u_prev_z = 0.0f;

    // Input matrix
    input = new Matrix(this, 4, 6, floatType, name);

    // Matrix descriptor for logging. It should be always a nx1 matrix.
    MatrixDescriptor *log_labels = new MatrixDescriptor(4, 1);
    log_labels->SetElementName(0, 0, "x_error");
    log_labels->SetElementName(1, 0, "y_error");
    log_labels->SetElementName(2, 0, "z_error");
    log_labels->SetElementName(3, 0, "mass_hat");
    state = new Matrix(this, log_labels, floatType, name);
    delete log_labels;

    // GUI for custom PID
    GroupBox *gui_customPID = new GroupBox(position, name);
    GroupBox *general_parameters = new GroupBox(gui_customPID->NewRow(), "General parameters");
    deltaT_custom = new DoubleSpinBox(general_parameters->NewRow(), "Custom dt [s]", 0, 1, 0.001, 4);
    k_motor = new DoubleSpinBox(general_parameters->LastRowLastCol(), "Motor constant", 0, 50, 0.01, 4, 29.5870);
    sat_pos = new DoubleSpinBox(general_parameters->NewRow(), "Saturation pos", 0, 10, 0.01, 3);
    sat_att = new DoubleSpinBox(general_parameters->LastRowLastCol(), "Saturation att", 0, 10, 0.01, 3);
    sat_thrust = new DoubleSpinBox(general_parameters->LastRowLastCol(), "Saturation thrust", 0, 10, 0.01, 3);
    
    // Neural network mass estimator parameters
    GroupBox *nn_group = new GroupBox(gui_customPID->NewRow(), "NN mass estimator");
    nn_hidden_neurons = new DoubleSpinBox(nn_group->NewRow(), "Hidden neurons", 1, 64, 1, 0, 6);
    nn_weight_std = new DoubleSpinBox(nn_group->LastRowLastCol(), "Weight std", 0, 1, 0.001, 4, 0.1);
    nn_learning_rate = new DoubleSpinBox(nn_group->NewRow(), "Learning rate", 0, 10, 0.0001, 4, 2.0);
    nn_regularization = new DoubleSpinBox(nn_group->LastRowLastCol(), "L2 regularization", 0, 0.1, 0.0001, 6, 0.0004);
    nn_eps0 = new DoubleSpinBox(nn_group->NewRow(), "Softplus eps0", 0, 1, 0.000001, 6, 0.001);
    nn_use_nlms = new DoubleSpinBox(nn_group->LastRowLastCol(), "Use NLMS (0/1)", 0, 1, 1, 0, 1);
    nn_u_nom = new DoubleSpinBox(nn_group->NewRow(), "u_{nom}", 0.1, 100, 0.1, 2, 10.0);
    nn_nu_nom = new DoubleSpinBox(nn_group->LastRowLastCol(), "nu_{nom}", 0.1, 100, 0.1, 2, 15.0);
    nn_mass_min = new DoubleSpinBox(nn_group->NewRow(), "Min mass [kg]", 0.1, 20, 0.01, 2, 0.1);
    nn_mass_max = new DoubleSpinBox(nn_group->LastRowLastCol(), "Max mass [kg]", 0.1, 20, 0.01, 2, 20.0);
    
    // Custom cartesian position controller
    GroupBox *custom_position = new GroupBox(gui_customPID->NewRow(), "Custom position controller");
    Kp_pos = new Vector3DSpinBox(custom_position->NewRow(), "Kp_pos", 0, 100, 0.1, 3);
    Kd_pos = new Vector3DSpinBox(custom_position->LastRowLastCol(), "Kd_pos", 0, 100, 0.1, 3);
    Ki_pos = new Vector3DSpinBox(custom_position->LastRowLastCol(), "Ki_pos", 0, 100, 0.1, 3);
    
    // Custom attitude controller
    GroupBox *custom_attitude = new GroupBox(gui_customPID->NewRow(), "Custom attitude controller");
    Kp_att = new Vector3DSpinBox(custom_attitude->NewRow(), "Kp_att", 0, 100, 0.1, 3);
    Kd_att = new Vector3DSpinBox(custom_attitude->LastRowLastCol(), "Kd_att", 0, 100, 0.1, 3);
    Ki_att = new Vector3DSpinBox(custom_attitude->LastRowLastCol(), "Ki_att", 0, 100, 0.1, 3);

    AddDataToLog(state);
    initializeNetwork();
}

MyController::~MyController()
{
    delete state;
}

void MyController::UpdateFrom(const io_data *data)
{
    ensureNetworkSize();

    float thrust = 0.0f;
    Vector3Df u, tau;

    if(deltaT_custom->Value() == 0)
    {
        delta_t = (float)(data->DataDeltaTime())/1000000000;
    }
    else
    {
        delta_t = deltaT_custom->Value();
    }

    if(first_update)
    {
        initial_time = double(GetTime())/1000000000;
        first_update = false;
    }

    // Obtain state
    input->GetMutex();
    Vector3Df pos_error(input->Value(0, 0), input->Value(1, 0), input->Value(2, 0));
    Vector3Df vel_error(input->Value(0, 1), input->Value(1, 1), input->Value(2, 1));
    float xppd = input->Value(0, 5);
    float yppd = input->Value(1, 5);
    float zppd = input->Value(2, 5);
    Quaternion q(input->Value(0, 2), input->Value(1, 2), input->Value(2, 2), input->Value(3, 2));
    Vector3Df omega(input->Value(0, 3), input->Value(1, 3), input->Value(2, 3));
    float yaw_ref = input->Value(0, 4);
    input->ReleaseMutex();

    // Get tunning parameters from GUI
    Vector3Df Kp_pos_val(Kp_pos->Value().x, Kp_pos->Value().y, Kp_pos->Value().z);
    Vector3Df Kd_pos_val(Kd_pos->Value().x, Kd_pos->Value().y, Kd_pos->Value().z);
    Vector3Df Ki_pos_val(Ki_pos->Value().x, Ki_pos->Value().y, Ki_pos->Value().z);
    Vector3Df Kp_att_val(Kp_att->Value().x, Kp_att->Value().y, Kp_att->Value().z);
    Vector3Df Kd_att_val(Kd_att->Value().x, Kd_att->Value().y, Kd_att->Value().z);
    Vector3Df Ki_att_val(Ki_att->Value().x, Ki_att->Value().y, Ki_att->Value().z);

    // Mass used for control
    float mass_for_control = clampMass(mass_hat);
    if(mass_for_control <= 0.0f)
    {
        mass_for_control = initialMassGuess();
        theta_hat = 1.0f / mass_for_control;
        mass_hat = mass_for_control;
    }

    float nux = xppd+Kp_pos_val.x*pos_error.x + Kd_pos_val.x*vel_error.x;
    float nuy = yppd+Kp_pos_val.y*pos_error.y + Kd_pos_val.y*vel_error.y;
    float nuz = zppd + Kp_pos_val.z*pos_error.z + Kd_pos_val.z*vel_error.z - g;

    // Cartesian custom controller
    u.x = mass_for_control*(nux);
    u.y = mass_for_control*(nuy);
    u.z = mass_for_control*(nuz);
    float ctrl_z = u.z; // This is the thrust needed to control the z position before saturation
    u.Saturate(sat_pos->Value());

    // Attitude custom controller
    Euler rpy = q.ToEuler();
    tau.x = Kp_att_val.x*(rpy.roll + u.y) + Kd_att_val.x*omega.x;
    tau.y = Kp_att_val.y*(rpy.pitch - u.x) + Kd_att_val.y*omega.y;
    tau.z = Kp_att_val.z*(rpy.YawDistanceFrom(yaw_ref)) + Kd_att_val.z*omega.z;
    applyMotorConstant(tau);
    tau.Saturate(sat_att->Value());

    // Compute custom thrust
    thrust = ctrl_z; // This is the thrust needed to counteract gravity and control the z position
    applyMotorConstant(thrust);
    float thr_sat = sat_thrust->Value();
    if(thrust < -thr_sat) {
      thrust = -thr_sat;
    } else if(thrust >= 0) {
        thrust = 0;
    }

    // Mass estimator update
    
    float actual_acc_z = zppd;
    if(has_prev_velocity && delta_t > 1e-6f)
    {
        float vel_error_rate = (vel_error.z - prev_vel_z) / delta_t;
        // actual_acc_z += zppd - vel_error_rate;
        actual_acc_z += vel_error_rate;
    }
    else
    {
        actual_acc_z = zppd;
    }
    if(!std::isfinite(actual_acc_z))
    {
        actual_acc_z = zppd;
    }
    prev_vel_z = vel_error.z;
    has_prev_velocity = true;
    float motor_const = static_cast<float>(k_motor->Value());
    if(motor_const < 1e-6f)
    {
        motor_const = 1e-6f;
    }
    float estimator_thrust = -thrust * motor_const;

    updateMassEstimator(delta_t, estimator_thrust, nuz, actual_acc_z);
    
    // Debug thrust value
    std::cout << " error_x: " << pos_error.x << " error_y: " << pos_error.y << " error_z: " << pos_error.z << " mass_hat: " << mass_hat <<std::endl;
    
    // Send controller output
    output->SetValue(0, 0, tau.x);
    output->SetValue(1, 0, tau.y);
    output->SetValue(2, 0, tau.z);
    output->SetValue(3, 0, thrust);
    output->SetDataTime(data->DataTime());
    
    // Log state (example).
    // Modify the log_labels matrix in the constructor to add more variables.
    state->GetMutex();
    state->SetValue(0, 0, pos_error.x);
    state->SetValue(1, 0, pos_error.y);
    state->SetValue(2, 0, pos_error.z);
    state->SetValue(3, 0, mass_hat);
    state->ReleaseMutex();

    ProcessUpdate(output);
}

void MyController::Reset(void)
{
    first_update = true;
    has_prev_velocity = false;
    prev_vel_z = 0.0f;
    u_prev_z = 0.0f;
    mass_hat = initialMassGuess();
    theta_hat = 1.0f / mass_hat;
}

void MyController::SetValues(Vector3Df pos_error, Vector3Df vel_error, Quaternion currentQuaternion, Vector3Df omega, float yaw_ref, float xppd, float yppd, float zppd)
{
    // Set the input values for the controller. 
    // This function is called from the main controller to set the input values.
    input->GetMutex();
    input->SetValue(0, 0, pos_error.x);
    input->SetValue(1, 0, pos_error.y);
    input->SetValue(2, 0, pos_error.z);

    input->SetValue(0, 1, vel_error.x);
    input->SetValue(1, 1, vel_error.y);
    input->SetValue(2, 1, vel_error.z);

    input->SetValue(0, 2, currentQuaternion.q0);
    input->SetValue(1, 2, currentQuaternion.q1);
    input->SetValue(2, 2, currentQuaternion.q2);
    input->SetValue(3, 2, currentQuaternion.q3);

    input->SetValue(0, 3, omega.x);
    input->SetValue(1, 3, omega.y);
    input->SetValue(2, 3, omega.z);

    // Set yaw reference
    input->SetValue(0, 4, yaw_ref);

    input->SetValue(0, 5, xppd);
    input->SetValue(1, 5, yppd);
    input->SetValue(2, 5, zppd);

    input->ReleaseMutex();
}

void MyController::applyMotorConstant(Vector3Df &signal)
{
    float motor_constant = k_motor->Value();
    signal.x = signal.x/motor_constant;
    signal.y = signal.y/motor_constant;
    signal.z = signal.z/motor_constant;
}

void MyController::applyMotorConstant(float &signal)
{
    float motor_constant = k_motor->Value();
    signal = signal/motor_constant;
}

void MyController::ensureNetworkSize(void)
{
    double hidden_value = nn_hidden_neurons->Value();
    if(hidden_value < 1.0)
    {
        hidden_value = 1.0;
    }
    size_t desired = static_cast<size_t>(hidden_value + 0.5);
    float current_std = static_cast<float>(nn_weight_std->Value());
    if(!network_ready || desired != hidden_neurons || std::fabs(current_std - last_weight_std) > 1e-6f)
    {
        initializeNetwork();
    }
}

void MyController::initializeNetwork(void)
{
    double hidden_value = nn_hidden_neurons->Value();
    if(hidden_value < 1.0)
    {
        hidden_value = 1.0;
    }
    hidden_neurons = static_cast<size_t>(hidden_value + 0.5);

    W1.assign(hidden_neurons, std::array<float, 2>{{0.0f, 0.0f}});
    b1.assign(hidden_neurons, 0.0f);
    w2.assign(hidden_neurons, 0.0f);
    hidden_layer.assign(hidden_neurons, 0.0f);

    std::normal_distribution<float> dist(0.0f, static_cast<float>(nn_weight_std->Value()));
    for(size_t i = 0; i < hidden_neurons; ++i)
    {
        for(size_t j = 0; j < 2; ++j)
        {
            W1[i][j] = dist(rng);
        }
        w2[i] = dist(rng);
    }

    b2 = 0.0f;
    last_weight_std = static_cast<float>(nn_weight_std->Value());

    mass_hat = initialMassGuess();
    theta_hat = 1.0f / mass_hat;
    prev_vel_z = 0.0f;
    u_prev_z = 0.0f;
    has_prev_velocity = false;
    network_ready = true;
}

void MyController::massBounds(float &min_mass, float &max_mass) const
{
    min_mass = static_cast<float>(nn_mass_min->Value());
    max_mass = static_cast<float>(nn_mass_max->Value());
    if(min_mass < 1e-3f)
    {
        min_mass = 1e-3f;
    }
    if(max_mass < min_mass + 1e-3f)
    {
        max_mass = min_mass + 1e-3f;
    }
}

float MyController::clampMass(float mass)
{
    float min_mass, max_mass;
    massBounds(min_mass, max_mass);
    mass = std::max(min_mass, std::min(max_mass, mass));
    return mass;
}

float MyController::initialMassGuess(void) const
{
    float min_mass, max_mass;
    massBounds(min_mass, max_mass);
    float guess = 0.5f * (min_mass + max_mass);
    if(guess <= 0.0f)
    {
        guess = std::max(min_mass, 1e-3f);
    }
    return std::max(min_mass, std::min(max_mass, guess));
}

float MyController::safeSoftplus(float x) const
{
    if(x > 20.0f)
    {
        return x;
    }
    if(x < -20.0f)
    {
        return std::exp(x);
    }
    return std::log1p(std::exp(x));
}

float MyController::sigmoid(float x) const
{
    if(x >= 0.0f)
    {
        float exp_neg = std::exp(-x);
        return 1.0f / (1.0f + exp_neg);
    }
    float exp_pos = std::exp(x);
    return exp_pos / (1.0f + exp_pos);
}

void MyController::updateMassEstimator(float dt, float thrust_input, float nuz, float actual_acc_z)
{
    if(!network_ready || hidden_neurons == 0)
    {
        u_prev_z = thrust_input;
        return;
    }

    float min_mass, max_mass;
    massBounds(min_mass, max_mass);
    float theta_min = 1.0f / max_mass;
    float theta_max = 1.0f / min_mass;

    float u_nom = std::max(static_cast<float>(nn_u_nom->Value()), 1e-6f);
    float nu_nom = std::max(static_cast<float>(nn_nu_nom->Value()), 1e-6f);
    float mu0 = u_prev_z / u_nom; // Normalized NN input
    float mu1 = nuz / nu_nom; // Normalized NN input 

    // float mu0 = u_prev_z; // Unnormalized NN input
    // float mu1 = nuz; // Unormalized NN input

    for(size_t i = 0; i < hidden_neurons; ++i)
    {
        float sum = W1[i][0] * mu0 + W1[i][1] * mu1 + b1[i];
        hidden_layer[i] = std::tanh(sum);
    }

    float z2 = b2;
    for(size_t i = 0; i < hidden_neurons; ++i)
    {
        z2 += w2[i] * hidden_layer[i];
    }

    float theta_raw = static_cast<float>(nn_eps0->Value()) + safeSoftplus(z2);
    theta_hat = std::max(theta_min, std::min(theta_raw, theta_max));
    
    float estimated_mass = 1.0f / theta_hat;
    if(!std::isfinite(estimated_mass))
    {
        estimated_mass = min_mass;
    }
    mass_hat = std::max(min_mass, std::min(max_mass, estimated_mass));

    float uz = thrust_input;
    float y_sup = actual_acc_z + g;
    float eps = y_sup - theta_hat * uz;

    float eta = static_cast<float>(nn_learning_rate->Value());
    if(eta > 0.0f && dt > 0.0f)
    {
        float lambda = static_cast<float>(nn_regularization->Value());
        float uz_sq = uz * uz;
        float g_norm;
        if(nn_use_nlms->Value() >= 0.5f)
        {
            g_norm = (uz_sq > 0.0f) ? (eps * uz) / (1.0f + uz_sq) : 0.0f;
        }
        else
        {
            g_norm = eps * uz;
        }
        
        float sig = sigmoid(z2);
        float delta_common = eta * g_norm * sig;

        for(size_t i = 0; i < hidden_neurons; ++i)
        {
            float h_val = hidden_layer[i];
            float dh = 1.0f - h_val * h_val;
            w2[i] += dt * (delta_common * h_val - lambda * w2[i]);
            float grad_common = delta_common * dh;
            W1[i][0] += dt * (grad_common * mu0 - lambda * W1[i][0]);
            W1[i][1] += dt * (grad_common * mu1 - lambda * W1[i][1]);
            b1[i] += dt * (grad_common - lambda * b1[i]);
        }
        b2 += dt * (delta_common - lambda * b2);
    }

    u_prev_z = uz;
}

void MyController::plotEstimatedMass(const LayoutPosition *position)
{
    DataPlot1D *mass_plot = new DataPlot1D(position, "Estimated Mass", 0, 4);
    mass_plot->AddCurve(state->Element(3), DataPlot::Blue);
}