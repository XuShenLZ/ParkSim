from dataclasses import dataclass, field

import numpy as np
import copy

@dataclass
class PythonMsg:
    def __setattr__(self,key,value):
        if not hasattr(self,key):
            raise TypeError ('Cannot add new field "%s" to frozen class %s' %(key,self))
        else:
            object.__setattr__(self,key,value)
    
    def pprint(self):
        print(self.print())
        
    def print(self, depth = 0, name = None):
        '''
        default __str__ method is not easy to read, especially for nested classes.
        This is easier to read but much longer
        
        Will not work with "from_str" method.
        '''
        print_str = ''
        for j in range(depth): print_str += '  '
        if name:
            print_str += name + ' (' + type(self).__name__ + '):\n'
        else:
            print_str += type(self).__name__ + ':\n'
        for key in vars(self):
            val = self.__getattribute__(key)
            if isinstance(val, PythonMsg):
                print_str += val.print(depth = depth + 1, name = key)
            else:
                for j in range(depth + 1): print_str += '  '
                print_str += str(key) + '=' + str(val)
                print_str += '\n'
        
        if depth == 0:
            print(print_str)
        else:
            return print_str
    
    
    def from_str(self,string_rep):
        '''
        inverts dataclass.__str__() method generated for this class so you can unpack objects sent via text (e.g. through multiprocessing.Queue)
        '''
        val_str_index = 0
        for key in vars(self):
            val_str_index = string_rep.find(key + '=', val_str_index) + len(key) + 1  #add 1 for the '=' sign
            value_substr  = string_rep[val_str_index : string_rep.find(',', val_str_index)]   #(thomasfork) - this should work as long as there are no string entries with commas

            if '\'' in value_substr:  # strings are put in quotes
                self.__setattr__(key, value_substr[1:-1])
            if 'None' in value_substr:
                self.__setattr__(key, None)
            else:
                self.__setattr__(key, float(value_substr))
    
    def copy(self):
        return copy.deepcopy(self)

@dataclass
class VehicleCoords(PythonMsg):
    '''
    Complete vehicle coordinates (local, global, and input)
    '''
    t: float  = field(default = None)    # time in seconds
    x: float = field(default = None)     # global x coordinate in meters
    y: float = field(default = None)     # global y coordinate in meters
    z: float = field(default = 0)        # global z coordinate in meters
    v: float = field(default = 0)        # speed in m/s
    v_x: float = field(default = None)   # global x velocity in m/s
    v_y: float = field(default = None)   # global y velocity in m/s
    a_x: float = field(default = None)   # global x acceleration in m/s^2
    a_y: float = field(default = None)   # global y acceleration in m/s^2
    psi: float = field(default = None)      # global vehicle heading angle
    psidot: float = field(default = None)   # global and local angular velocity of car
    v_long: float = field(default = None)   # longitudinal velocity (in the direction of psi)
    v_tran: float = field(default = None)   # transverse velocity   (orthogonal to the direction of psi)
    a_long: float = field(default = None)   # longitudinal velocity (in the direction of psi)
    a_tran: float = field(default = None)   # transverse velocity   (orthogonal to the direction of psi)

    e_psi: float = field(default = None)    # heading error between car and track
    s: float = field(default = None)        # path length along center of track to projected position of car
    x_tran: float = field(default = None)   # deviation from centerline (transverse position)
    u_a: float = field(default = None)      # acceleration output
    u_steer: float = field(default = None)  # steering angle output

    def __str__(self):
        return 't:{self.t}, x:{self.x}, y:{self.y}, psi:{self.psi}, v_long:{self.v_long}, v_tran:{self.v_tran}, psidot:{self.psidot}, u_a:{self.u_a}, u_steer:{self.u_steer}'.format(self=self)

    def get_R(self, reverse = False):
        # Warning - Not suitable for general 3D case
        psi = self.psi
        return np.array([[np.cos(psi), -np.sin(psi),0],[np.sin(psi),np.cos(psi),0],[0,0,0]])

    def L2dist(self, p: 'VehicleCoords'):
        # Calculate the L2 distance between the vehicle centers
        return np.sqrt((self.x - p.x)**2 + (self.y - p.y)**2)

    def update_body_velocity_from_global(self):
        self.v_long =  self.v_x * np.cos(self.psi) + self.v_y * np.sin(self.psi)
        self.v_tran = -self.v_x * np.sin(self.psi) + self.v_y * np.cos(self.psi)
        self.a_long =  self.a_x * np.cos(self.psi) + self.a_y * np.sin(self.psi)
        self.a_tran = -self.a_y * np.sin(self.psi) + self.a_y + np.cos(self.psi)

    def update_global_velocity_from_body(self):
        self.v_x =  self.v_long * np.cos(self.psi) - self.v_tran * np.sin(self.psi)
        self.v_y =  self.v_long * np.sin(self.psi) + self.v_tran * np.cos(self.psi)
        self.a_x =  self.a_long * np.cos(self.psi) - self.a_tran * np.sin(self.psi)
        self.a_y =  self.a_long * np.sin(self.psi) + self.a_tran * np.cos(self.psi)

    def update_global_velocity_from_speed(self):
        self.v_x = self.v * np.cos(self.psi)
        self.v_y = self.v * np.sin(self.psi)