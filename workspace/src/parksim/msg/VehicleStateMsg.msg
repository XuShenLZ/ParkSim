# This is a message to hold data with system state information
std_msgs/Header header
float64                                  t       # Time

parksim/PositionMsg                x

parksim/OrientationEulerMsg        e # Orientation in Euler angles
parksim/OrientationQuaternionMsg   q # Orientation in quaternions
parksim/BodyAngularVelocityMsg     w # Vector of angular velocities
parksim/BodyAngularAccelerationMsg aa # Vector of angular accelerations
parksim/BodyLinearVelocityMsg      v # local body velocities
parksim/BodyLinearAccelerationMsg  a # local body accelerations

parksim/ParametricPoseMsg          p
parksim/ParametricVelocityMsg      pt

parksim/VehicleActuationMsg        u

float64                                  lap_num    # Lap number
