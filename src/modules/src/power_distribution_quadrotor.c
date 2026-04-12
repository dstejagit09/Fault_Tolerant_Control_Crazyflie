/**
 *    ||          ____  _ __
 * +------+      / __ )(_) /_______________ _____  ___
 * | 0xBC |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
 * +------+    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
 *  ||  ||    /_____/_/\__/\___/_/   \__,_/ /___/\___/
 *
 * Crazyflie control firmware
 *
 * Copyright (C) 2011-2022 Bitcraze AB
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, in version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * power_distribution_quadrotor.c - Crazyflie stock power distribution code
 */


#include "power_distribution.h"

#include <string.h>
#include "debug.h"
#include "log.h"
#include "param.h"
#include "num.h"
#include "autoconf.h"
#include "config.h"
#include "math.h"
#include "platform_defaults.h"

#if (!defined(CONFIG_MOTORS_REQUIRE_ARMING) || (CONFIG_MOTORS_REQUIRE_ARMING == 0)) && defined(CONFIG_MOTORS_DEFAULT_IDLE_THRUST) && (CONFIG_MOTORS_DEFAULT_IDLE_THRUST > 0)
    #error "CONFIG_MOTORS_REQUIRE_ARMING must be defined and not set to 0 if CONFIG_MOTORS_DEFAULT_IDLE_THRUST is greater than 0"
#endif
#ifndef CONFIG_MOTORS_DEFAULT_IDLE_THRUST
#  define DEFAULT_IDLE_THRUST 0
#else
#  define DEFAULT_IDLE_THRUST CONFIG_MOTORS_DEFAULT_IDLE_THRUST
#endif

static uint32_t idleThrust = DEFAULT_IDLE_THRUST;

// Fault-tolerant 3-motor allocation
// failedMotor: 0 = none (normal 4-motor), 1-4 = which motor has failed
static uint8_t failedMotor = 0;
static float residualYaw = 0.0f;
static uint8_t ftActive = 0;

/**
 * @brief 3-motor closed-form allocation when one motor has failed.
 *
 * Prioritizes exact tracking of thrust, roll, and pitch.
 * Yaw becomes a residual determined by the 3 surviving motors.
 *
 * Derived from the reduced allocation matrix A_i (4x3) for each
 * failed motor case i, solving the 3x3 system for (T, tau_x, tau_y)
 * in firmware-scaled variables where r = roll/2, p = pitch/2.
 */
static void powerDistributionFaultTolerant(const control_t *control, motors_thrust_uncapped_t* motorThrustUncapped)
{
  float T = control->thrust;
  float r = control->roll / 2.0f;
  float p = control->pitch / 2.0f;

  // Zero all motors first
  motorThrustUncapped->motors.m1 = 0;
  motorThrustUncapped->motors.m2 = 0;
  motorThrustUncapped->motors.m3 = 0;
  motorThrustUncapped->motors.m4 = 0;

  ftActive = 1;

  switch (failedMotor) {
    case 1:
      // Motor 1 dead. Working: 2, 3, 4
      // m2 = 2(T - r), m3 = 2(r - p), m4 = 2(T + p)
      motorThrustUncapped->motors.m2 = 2.0f * (T - r);
      motorThrustUncapped->motors.m3 = 2.0f * (r - p);
      motorThrustUncapped->motors.m4 = 2.0f * (T + p);
      residualYaw = -T + r - p;
      break;

    case 2:
      // Motor 2 dead. Working: 1, 3, 4
      // m1 = 2(T - r), m3 = 2(T - p), m4 = 2(r + p)
      motorThrustUncapped->motors.m1 = 2.0f * (T - r);
      motorThrustUncapped->motors.m3 = 2.0f * (T - p);
      motorThrustUncapped->motors.m4 = 2.0f * (r + p);
      residualYaw = T - r - p;
      break;

    case 3:
      // Motor 3 dead. Working: 1, 2, 4
      // m1 = 2(p - r), m2 = 2(T - p), m4 = 2(T + r)
      motorThrustUncapped->motors.m1 = 2.0f * (p - r);
      motorThrustUncapped->motors.m2 = 2.0f * (T - p);
      motorThrustUncapped->motors.m4 = 2.0f * (T + r);
      residualYaw = p - r - T;
      break;

    case 4:
      // Motor 4 dead. Working: 1, 2, 3
      // m1 = 2(T + p), m2 = -2(r + p), m3 = 2(T + r)
      motorThrustUncapped->motors.m1 = 2.0f * (T + p);
      motorThrustUncapped->motors.m2 = -2.0f * (r + p);
      motorThrustUncapped->motors.m3 = 2.0f * (T + r);
      residualYaw = T + r + p;
      break;

    default:
      // No fault or invalid, fall back to normal 4-motor
      ftActive = 0;
      motorThrustUncapped->motors.m1 = T - r + p + control->yaw;
      motorThrustUncapped->motors.m2 = T - r - p - control->yaw;
      motorThrustUncapped->motors.m3 = T + r - p + control->yaw;
      motorThrustUncapped->motors.m4 = T + r + p - control->yaw;
      residualYaw = control->yaw;
      break;
  }
}

int powerDistributionMotorType(uint32_t id)
{
  return 1;
}

uint16_t powerDistributionStopRatio(uint32_t id)
{
  return 0;
}

void powerDistributionInit(void)
{
  #if (!defined(CONFIG_MOTORS_REQUIRE_ARMING) || (CONFIG_MOTORS_REQUIRE_ARMING == 0))
  if(idleThrust > 0) {
    DEBUG_PRINT("WARNING: idle thrust will be overridden with value 0. Autoarming can not be on while idle thrust is higher than 0. If you want to use idle thust please use use arming\n");
  }
  #endif
}

bool powerDistributionTest(void)
{
  bool pass = true;
  return pass;
}

static uint16_t capMinThrust(float thrust, uint32_t minThrust) {
  if (thrust < minThrust) {
    return minThrust;
  }

  return thrust;
}

static void powerDistributionLegacy(const control_t *control, motors_thrust_uncapped_t* motorThrustUncapped)
{
  int16_t r = control->roll / 2.0f;
  int16_t p = control->pitch / 2.0f;

  motorThrustUncapped->motors.m1 = control->thrust - r + p + control->yaw;
  motorThrustUncapped->motors.m2 = control->thrust - r - p - control->yaw;
  motorThrustUncapped->motors.m3 = control->thrust + r - p + control->yaw;
  motorThrustUncapped->motors.m4 = control->thrust + r + p - control->yaw;
}

static void powerDistributionForceTorque(const control_t *control, motors_thrust_uncapped_t* motorThrustUncapped) {
  static float motorForces[STABILIZER_NR_OF_MOTORS];

  const float arm = 0.707106781f * ARM_LENGTH;
  const float rollPart = 0.25f / arm * control->torqueX;
  const float pitchPart = 0.25f / arm * control->torqueY;
  const float thrustPart = 0.25f * control->thrustSi; // N (per rotor)
  const float yawPart = 0.25f * control->torqueZ / THRUST2TORQUE;

  motorForces[0] = thrustPart - rollPart - pitchPart - yawPart;
  motorForces[1] = thrustPart - rollPart + pitchPart + yawPart;
  motorForces[2] = thrustPart + rollPart + pitchPart - yawPart;
  motorForces[3] = thrustPart + rollPart - pitchPart + yawPart;

  for (int motorIndex = 0; motorIndex < STABILIZER_NR_OF_MOTORS; motorIndex++) {
    float motorForce = motorForces[motorIndex];
    if (motorForce < 0.0f) {
      motorForce = 0.0f;
    }

    motorThrustUncapped->list[motorIndex] = motorForce / THRUST_MAX * UINT16_MAX;
  }
}

/**
 * @brief Allows for direct control of motor power with clipping
 *
 * This function applies clipping to the motor values, which is different to
 * the "capping" behaviour found in powerDistributionForceTorque() - which
 * instead prioritizes stability rather than thrust.
 */
static void powerDistributionForce(const control_t *control, motors_thrust_uncapped_t* motorThrustUncapped) {
  for (int i = 0; i < STABILIZER_NR_OF_MOTORS; i++) {
    float f = control->normalizedForces[i];

    if (f < 0.0f) {
      f = 0.0f;
    }

    if (f > 1.0f) {
      f = 1.0f;
    }

    motorThrustUncapped->list[i] = f * UINT16_MAX;
  }
}

void powerDistribution(const control_t *control, motors_thrust_uncapped_t* motorThrustUncapped)
{
  // If a motor fault is active, override with 3-motor allocation
  if (failedMotor >= 1 && failedMotor <= 4) {
    powerDistributionFaultTolerant(control, motorThrustUncapped);
    return;
  }

  switch (control->controlMode) {
    case controlModeLegacy:
      powerDistributionLegacy(control, motorThrustUncapped);
      break;
    case controlModeForceTorque:
      powerDistributionForceTorque(control, motorThrustUncapped);
      break;
    case controlModeForce:
      powerDistributionForce(control, motorThrustUncapped);
      break;
    default:
      // Nothing here
      break;
  }
}

bool powerDistributionCap(const motors_thrust_uncapped_t* motorThrustBatCompUncapped, motors_thrust_pwm_t* motorPwm)
{
  const int32_t maxAllowedThrust = UINT16_MAX;
  bool isCapped = false;

  // Find highest thrust
  int32_t highestThrustFound = 0;
  for (int motorIndex = 0; motorIndex < STABILIZER_NR_OF_MOTORS; motorIndex++)
  {
    const int32_t thrust = motorThrustBatCompUncapped->list[motorIndex];
    if (thrust > highestThrustFound)
    {
      highestThrustFound = thrust;
    }
  }

  int32_t reduction = 0;
  if (highestThrustFound > maxAllowedThrust)
  {
    reduction = highestThrustFound - maxAllowedThrust;
    isCapped = true;
  }

  for (int motorIndex = 0; motorIndex < STABILIZER_NR_OF_MOTORS; motorIndex++)
  {
    int32_t thrustCappedUpper = motorThrustBatCompUncapped->list[motorIndex] - reduction;
    motorPwm->list[motorIndex] = capMinThrust(thrustCappedUpper, powerDistributionGetIdleThrust());
  }

  return isCapped;
}

uint32_t powerDistributionGetIdleThrust()
{
  int32_t thrust = idleThrust;
  #if (!defined(CONFIG_MOTORS_REQUIRE_ARMING) || (CONFIG_MOTORS_REQUIRE_ARMING == 0))
    thrust = 0;
  #endif
  return thrust;
}

float powerDistributionGetMaxThrust() {
  return STABILIZER_NR_OF_MOTORS * THRUST_MAX;
}

/**
 * Power distribution parameters
 */
PARAM_GROUP_START(powerDist)
/**
 * @brief Motor thrust to set at idle (default: 0)
 *
 * This is often needed for brushless motors as
 * it takes time to start up the motor. Then a
 * common value is between 3000 - 6000.
 */
PARAM_ADD_CORE(PARAM_UINT32 | PARAM_PERSISTENT, idleThrust, &idleThrust)
/**
 * @brief Which motor has failed (0 = none, 1-4 = failed motor index)
 *
 * Setting this to a value 1-4 activates the fault-tolerant 3-motor
 * closed-form allocation, which prioritizes thrust, roll, and pitch
 * while letting yaw be a residual.
 */
PARAM_ADD_CORE(PARAM_UINT8, failedMotor, &failedMotor)
PARAM_GROUP_STOP(powerDist)

/**
 * Logging for fault-tolerant power distribution
 */
LOG_GROUP_START(ftAlloc)
/**
 * @brief Whether fault-tolerant allocation is active
 */
LOG_ADD(LOG_UINT8, active, &ftActive)
/**
 * @brief Which motor is marked as failed
 */
LOG_ADD(LOG_UINT8, failedMotor, &failedMotor)
/**
 * @brief Residual yaw torque from 3-motor allocation
 */
LOG_ADD(LOG_FLOAT, residualYaw, &residualYaw)
LOG_GROUP_STOP(ftAlloc)
