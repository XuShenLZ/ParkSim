# This is a message to hold data with auxlary system information
std_msgs/Header header

parksim/PredictionMsg               ref_pose    # Reference Pose
float64                             ref_v   # Reference Velocity
int16                               target_idx  # Target Index
int16                               priority    # priority
string                              parking_flag    # Parking Flag
string                              parking_process # Parking Process (is currently parking)
bool                                is_braking  # Is braking
float64                             parking_start_time  # (Un)Parking Start Time
int16                               waiting_for # The index that vehicle is waiting for
string                              disp_text # Some custom text to displayed on top of vehicle