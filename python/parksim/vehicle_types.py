from dataclasses import dataclass, field
import numpy as np

from parksim.pytypes import *
from parksim.obstacle_types import BasePolytopeObstacle

@dataclass
class VehicleBody(BasePolytopeObstacle):
    """
    Class to represent the body of a rectangular vehicle in the body frame
    
    verticies everywhere are computed with the assumption that 0 degrees has the vehicle pointing east.
    matrices are computed for 0 degrees, with the assumption that they are rotated by separate code.
    """
    
    # dimensions of the stock rover
    vehicle_flag: int = field(default = 0)  # for different types of vehicles
    # Wheelbase
    lf: float = field(default = 0)
    lr: float = field(default = 0)
    bf: float = field(default = 0)
    br: float = field(default = 0)
    wb: float = field(default = 0)
    # Total Length and width
    l: float = field(default = 0)
    w : float = field(default = 0)
    h:  float = field(default = 0)

    # Wheel diameter and width
    wheel_d: float = field(default = 0.72)
    wheel_w: float = field(default = 0.22)

    # Circle Approximation
    cr: float = field(default = 0) # Offset of the first circle center in front
    cf: float = field(default = 0) # Offset of the first circle center at rear
    num_circles: int = field(default = 3)
    
    def __post_init__(self):
        if self.vehicle_flag == 0:
            self.lr = 1.35
            self.lf = 1.35
            self.wb = self.lr + self.lf
            self.w = 1.85
            self.bf = 2.3
            self.br = 2.3
            self.l = self.br + self.bf

            self.cf = 2.25
            self.cr = 2.25
            self.num_circles = 5
        else:
            raise NotImplementedError('Unrecognized vehicle flag: %d'%self.vehicle_flag)
    
        self.__calc_V__()
        self.__calc_A_b__()
        return
        
    def __calc_V__(self):
        xy = np.array([[self.l/2, self.w/2],
                       [-self.l/2, self.w/2],
                       [-self.l/2, -self.w/2],
                       [self.l/2, -self.w/2],
                       [self.l/2, self.w/2]])
        
        V = xy[:-1,:]
                           
        object.__setattr__(self,'xy',xy)
        object.__setattr__(self,'V',V  )
        return
        
    def __calc_A_b__(self):
        A = np.array([[1,0],
                      [0,1],
                      [-1,0],
                      [0,-1]])
        
        b = np.array([self.l/2, self.w/2, self.l/2, self.w/2])
        object.__setattr__(self,'A',A)
        object.__setattr__(self,'b',b)
        return
    

@dataclass
class VehicleConfig(PythonMsg):
    '''
    vehicle configuration class
    for dt and all vehicle limits
    '''
    dt: float = field(default=0.1)  # vehicle simulation time step (applies to rest of vehicle as well)
    M: int = field(default = 4) # RK4 steps per interval

    # Vehicle Limits
    v_max: float = field(default=5)  # maximum velocity
    v_min: float = field(default=-5)  # minimum velocity
    a_max: float = field(default=10)  # maximum acceleration
    a_min: float = field(default=-10)  # minimum acceleration
    a_max_parking: float = field(default=2)  # maximum acceleration for parking
    a_min_parking: float = field(default=-2)  # minimum acceleration for parking
    delta_max: float = field(default=np.deg2rad(40.0))  # maximum steering angle
    delta_min: float = field(default=-np.deg2rad(40.0))  # minimum steering angle
    d_delta_max: float = field(default=1.5)  # maximum change in steering angle over dt
    d_delta_min: float = field(default=-1.5)  # minimum change in steering angle over dt

    # Path Following Related
    v_cruise: float = field(default=5) # Reference cruising speed
    steps_to_end: int = field(default=30) # The steps towards the end of the ref path to slow down
    v_end: float = field(default=1) # The speed in the final segment of tracking

    # Decision Making Related
    offset: float = field(default=1.75) # distance off from waypoints
    parking_start_offset: float = field(default=1.75) # distance off from waypoints to start parking maneuver
    look_ahead_timesteps: int = field(default=10) # how far to look ahead for crash detection
    crash_check_radius: float = field(default=15) # which vehicles to check crash
    braking_distance: float = field(default=10)
    parking_radius: float = field(default=9) # how much room a vehicle should have to park
    parking_ahead_angle: float = field(default=np.pi/4)
    leading_trailing_thres: float = field(default=0.25) # Threshold of heading angle difference to check whether two vehicles are leading and trailing. 0.25 is about about 15 degrees

@dataclass
class VehicleInfo(PythonMsg):
    ref_pose: VehiclePrediction = field(default=None)
    ref_v: float = field(default=0)
    target_idx: int = field(default=None)
    priority: int = field(default=None)
    task: str = field(default=None)
    parking_progress: str = field(default=None)
    is_braking: bool = field(default=None)
    parking_start_time: float = field(default=None)
    waiting_for: int = field(default=None)
    disp_text: str = field(default=None)
    is_all_done: bool = field(default=None)

    def __post_init__(self):
        self.ref_pose = VehiclePrediction()

@dataclass
class VehicleTask(PythonMsg):
    name: str = field(default=None) # Can be "CRUISE", "PARK", "UNPARK", "IDLE"
    v_cruise: float = field(default=None)
    target_spot_index: int = field(default=None)
    target_coords: np.ndarray = field(default=None)
    duration: float = field(default=None)
    end_time: float = field(default=None)
