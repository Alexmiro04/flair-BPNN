#ifndef MYCTRL_H
#define MYCTRL_H

#include <Object.h>
#include <ControlLaw.h>
#include <Vector3D.h>
#include <Quaternion.h>
#include <array>
#include <random>
#include <string>
#include <vector>

namespace flair {
    namespace core {
        class Matrix;
        class io_data;
    }
    namespace gui {
        class LayoutPosition;
        class DoubleSpinBox;
        class CheckBox;
        class Label;
        class Vector3DSpinBox;
    }
    namespace filter {
        // If you prefer to use a custom controller class, you can define it here.
        // ...
    }
}

namespace flair {
    namespace filter {
        class MyController : public ControlLaw
        {
            public :
                MyController(const flair::gui::LayoutPosition *position, const std::string &name);
                ~MyController();
                void UpdateFrom(const flair::core::io_data *data);
                void Reset(void);
                void SetValues(flair::core::Vector3Df pos_error, flair::core::Vector3Df vel_error, flair::core::Quaternion currentQuaternion, flair::core::Vector3Df omega, float yaw_ref, float xppd, float yppd, float zppd);
                void applyMotorConstant(flair::core::Vector3Df &signal);
                void applyMotorConstant(float &signal);
                void plotEstimatedMass(const flair::gui::LayoutPosition *position);

            private :
                float delta_t, initial_time;
                float g = 9.81;
                bool first_update;
                bool has_prev_velocity;

                static constexpr size_t kStateBaseEntries = 5;
                static constexpr size_t kLoggedNeurons = 1;
                static constexpr size_t kWeightsPerLoggedNeuron = 4;

                flair::core::Matrix *state;
                flair::gui::Vector3DSpinBox *Kp_pos, *Kd_pos, *Ki_pos, *Kp_att, *Kd_att, *Ki_att;
                flair::gui::DoubleSpinBox *deltaT_custom, *k_motor, *sat_pos, *sat_att, *sat_thrust;
                flair::gui::DoubleSpinBox *nn_hidden_neurons, *nn_weight_std, *nn_learning_rate, *nn_regularization;
                flair::gui::DoubleSpinBox *nn_eps0, *nn_u_nom, *nn_nu_nom, *nn_mass_min, *nn_mass_max, *nn_use_nlms;

                std::mt19937 rng;
                bool network_ready;
                size_t hidden_neurons;
                float last_weight_std;
                std::vector<std::array<float, 2>> W1;
                std::vector<float> b1;
                std::vector<float> w2;
                std::vector<float> hidden_layer;
                float b2;
                float theta_hat;
                float mass_hat;
                float prev_vel_z;
                float u_prev_z;

                void ensureNetworkSize(void);
                void initializeNetwork(void);
                void updateMassEstimator(float dt, float thrust_input, float nuz, float actual_acc_z);
                float clampMass(float mass);
                void massBounds(float &min_mass, float &max_mass) const;
                float initialMassGuess(void) const;
                float safeSoftplus(float x) const;
                float sigmoid(float x) const;
        };
    }
}

#endif // MYCTRL_H