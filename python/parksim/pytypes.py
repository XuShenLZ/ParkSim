from dataclasses import dataclass, field
import time
import array
import numpy as np
import pdb
import copy
import matplotlib.pyplot as plt

#DEFAULT_VEHICLE_TYPE = 'barc'

@dataclass
class PythonMsg:
    '''
    Base class for python messages. Intention is that fields cannot be changed accidentally,
    e.g. if you try "state.xk = 10", it will throw an error if no field "xk" exists.
    This helps avoid typos. If you really need to add a field use "object.__setattr__(state,'xk',10)"
    Dataclasses autogenerate a constructor, e.g. a dataclass fields x,y,z can be instantiated
    "pos = Position(x = 1, y = 2, z = 3)" without you having to write the __init__() function, just add the decorator
    Together it is hoped that these provide useful tools and safety measures for storing and moving data around by name,
    rather than by magic indices in an array, e.g. q = [9, 4.5, 8829] vs. q.x = 10, q.y = 9, q.z = 16
    '''
    def __setattr__(self, key, value):
        '''
        Overloads default attribute-setting functionality to avoid creating new fields that don't already exist
        This exists to avoid hard-to-debug errors from accidentally adding new fields instead of modifying existing ones

        To avoid this, use:
        object.__setattr__(instance, key, value)
        ONLY when absolutely necessary.
        '''
        if not hasattr(self, key):
            raise TypeError('Cannot add new field "%s" to frozen class %s' % (key, self))
        else:
            object.__setattr__(self, key, value)

    def print(self, depth=0, name=None):
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
                print_str += val.print(depth=depth + 1, name=key)
            else:
                for j in range(depth + 1): print_str += '  '
                print_str += str(key) + '=' + str(val)
                print_str += '\n'

        if depth == 0:
            print(print_str)
        else:
            return print_str

    def from_str(self, string_rep):
        '''
        inverts dataclass.__str__() method generated for this class so you can unpack objects sent via text (e.g. through multiprocessing.Queue)
        '''
        val_str_index = 0
        for key in vars(self):
            val_str_index = string_rep.find(key + '=', val_str_index) + len(key) + 1  # add 1 for the '=' sign
            value_substr = string_rep[val_str_index: string_rep.find(',',
                                                                     val_str_index)]  # (thomasfork) - this should work as long as there are no string entries with commas

            if '\'' in value_substr:  # strings are put in quotes
                self.__setattr__(key, value_substr[1:-1])
            if 'None' in value_substr:
                self.__setattr__(key, None)
            else:
                self.__setattr__(key, float(value_substr))

    def copy(self):
        return copy.deepcopy(self)

@dataclass
class NodeParamTemplate:
    '''
    Base class for node parameter templates
    used by autodeclaring and loading parameters as implemented in mpclab_base_nodes.py
    This class provides functionality for turning a generated template into a default configuration file (.yaml format)
    This itself is not a template. To create a template create a class (preferably in the source code for the node itself)
    and in __init__() add attributes to the class, e.g. self.dt = 0.1, self.state = VehicleCoords(), etc...
    These must be added in __init__(), not outside of it, or the variables wont show up in vars(instace_of_template).
    '''

    def spew_yaml(self):
        def append_param(yaml_str, key, val, indent_depth):
            for j in range(indent_depth):
                yaml_str += '  '

            yaml_str += key
            yaml_str += ': '
            if isinstance(val, str):
                yaml_str += "'" + val + "'"
            elif isinstance(val, np.ndarray):
                yaml_str += val.tolist().__str__()
            elif isinstance(val, (bool, int, float, str)):
                yaml_str += val.__str__()
            elif isinstance(val, (list, tuple, array.array)):
                yaml_str += val.__str__()
            elif val is None:
                yaml_str += ''
            else:
                yaml_str += '0'

            yaml_str += '\n'
            return yaml_str


        def unpack_pythonmsg(yaml_str, msg, prefix, depth = 2):
            yaml_str = append_param(yaml_str, prefix, None, depth)
            for key in vars(msg):
                val = msg.__getattribute__(key)
                if isinstance(val, PythonMsg):
                    yaml_str = unpack_pythonmsg(yaml_str, val, key, depth + 1)
                else:
                    yaml_str = append_param(yaml_str, key, val, depth + 1)


            return yaml_str

        yaml_str = 'namespace:\n'
        yaml_str += '  ros__parameters:\n'

        parameters = []
        for key in vars(self):
            val = self.__getattribute__(key)
            if isinstance(val, PythonMsg):
                yaml_str = unpack_pythonmsg(yaml_str, val, key)
            else:
                yaml_str = append_param(yaml_str, key, val, 2)

        return yaml_str

@dataclass
class Position(PythonMsg):
    x: float = field(default=0)
    y: float = field(default=0)
    z: float = field(default=0)

    # def xdot(self, q: 'OrientationQuaternion', v: 'BodyLinearVelocity') -> 'Position':
    #     xdot = Position()
    #     xdot.x = (1 - 2 * q.qj ** 2 - 2 * q.qk ** 2) * v.v1 + 2 * (q.qi * q.qj - q.qk * q.qr) * v.v2 + 2 * (
    #                 q.qi * q.qk + q.qj * q.qr) * v.v3
    #     xdot.y = (1 - 2 * q.qk ** 2 - 2 * q.qi ** 2) * v.v2 + 2 * (q.qj * q.qk - q.qi * q.qr) * v.v3 + 2 * (
    #                 q.qj * q.qi + q.qk * q.qr) * v.v1
    #     xdot.z = (1 - 2 * q.qi ** 2 - 2 * q.qj ** 2) * v.v3 + 2 * (q.qk * q.qi - q.qj * q.qr) * v.v1 + 2 * (
    #                 q.qk * q.qj + q.qi * q.qr) * v.v2
    #     return xdot

@dataclass
class VehicleActuation(PythonMsg):
    t: float = field(default = 0)

    u_a: float = field(default = 0)
    u_steer: float = field(default = 0)

    def __str__(self):
        return 't:{self.t}, u_a:{self.u_a}, u_steer:{self.u_steer}'.format(self=self)


@dataclass
class BodyLinearVelocity(PythonMsg):
    v_long: float = field(default=0)
    v_tran: float = field(default=0)
    v_n: float = field(default=0)
    v: float = field(default=0) # Speed for bicycle model

    def mag(self):
        return np.sqrt(self.v_long**2 + self.v_tran**2 + self.v_n**2)


@dataclass
class BodyAngularVelocity(PythonMsg):
    w_phi: float = field(default=0)
    w_theta: float = field(default=0)
    w_psi: float = field(default=0)


@dataclass
class BodyLinearAcceleration(PythonMsg):
    a_long: float = field(default=0)
    a_tran: float = field(default=0)
    a_n: float = field(default=0)


@dataclass
class BodyAngularAcceleration(PythonMsg):
    a_phi: float = field(default=0)
    a_theta: float = field(default=0)
    a_psi: float = field(default=0)

@dataclass
class OrientationEuler(PythonMsg):
    phi: float = field(default=0)
    theta: float = field(default=0)
    psi: float = field(default=0)

@dataclass
class OrientationQuaternion(PythonMsg):
    qr: float = field(default=1)
    qi: float = field(default=0)
    qj: float = field(default=0)
    qk: float = field(default=0)

    def e1(self):
        return np.array([[1 - 2 * self.qj ** 2 - 2 * self.qk ** 2,
                          2 * (self.qi * self.qj + self.qk * self.qr),
                          2 * (self.qi * self.qk - self.qj * self.qr)]]).T

    def e2(self):
        return np.array([[2 * (self.qi * self.qj - self.qk * self.qr),
                          1 - 2 * self.qi ** 2 - 2 * self.qk ** 2,
                          2 * (self.qj * self.qk + self.qi * self.qr)]]).T

    def e3(self):
        return np.array([[2 * (self.qi * self.qk + self.qj * self.qr),
                          2 * (self.qj * self.qk - self.qi * self.qr),
                          1 - 2 * self.qi ** 2 - 2 * self.qj ** 2]]).T

    def R(self):
        return np.array([[1 - 2 * self.qj ** 2 - 2 * self.qk ** 2, 2 * (self.qi * self.qj - self.qk * self.qr),
                          2 * (self.qi * self.qk + self.qj * self.qr)],
                         [2 * (self.qi * self.qj + self.qk * self.qr), 1 - 2 * self.qi ** 2 - 2 * self.qk ** 2,
                          2 * (self.qj * self.qk - self.qi * self.qr)],
                         [2 * (self.qi * self.qk - self.qj * self.qr), 2 * (self.qj * self.qk + self.qi * self.qr),
                          1 - 2 * self.qi ** 2 - 2 * self.qj ** 2]])

    def Rinv(self):
        return np.array([[1 - 2 * self.qj ** 2 - 2 * self.qk ** 2, 2 * (self.qi * self.qj + self.qk * self.qr),
                          2 * (self.qi * self.qk - self.qj * self.qr)],
                         [2 * (self.qi * self.qj - self.qk * self.qr), 1 - 2 * self.qi ** 2 - 2 * self.qk ** 2,
                          2 * (self.qj * self.qk + self.qi * self.qr)],
                         [2 * (self.qi * self.qk + self.qj * self.qr), 2 * (self.qj * self.qk - self.qi * self.qr),
                          1 - 2 * self.qi ** 2 - 2 * self.qj ** 2]])

    def norm(self):
        return np.sqrt(self.qr ** 2 + self.qi ** 2 + self.qj ** 2 + self.qk ** 2)

    def normalize(self):
        norm = self.norm()

        self.qr /= norm
        self.qi /= norm
        self.qj /= norm
        self.qk /= norm
        return

    def from_yaw(self, yaw):
        self.qi = 0
        self.qj = 0
        self.qr = np.cos(yaw / 2)
        self.qk = np.sin(yaw / 2)
        self.normalize()

        if abs(yaw - self.to_yaw()) % (2 * np.pi) > 1e-9: pdb.set_trace()

        return

    def to_yaw(self):
        return 2 * np.arctan2(self.qk, self.qr)

    def qdot(self, w: BodyAngularVelocity) -> 'OrientationQuaternion':
        qdot = OrientationQuaternion()
        qdot.qr = -0.5 * (self.qi * w.w_phi + self.qj * w.w_theta + self.qk * w.w_psi)
        qdot.qi = 0.5 * (self.qr * w.w_phi + self.qj * w.w_psi - self.qk * w.w_theta)
        qdot.qj = 0.5 * (self.qr * w.w_theta + self.qk * w.w_phi - self.qi * w.w_psi)
        qdot.qk = 0.5 * (self.qr * w.w_psi + self.qi * w.w_theta - self.qj * w.w_phi)
        return qdot

@dataclass
class ParametricPose(PythonMsg):
    s: float = field(default = 0)
    x_tran: float = field(default = 0)
    n: float = field(default = 0)
    e_psi: float = field(default = 0)

@dataclass
class ParametricVelocity(PythonMsg):
    ds: float = field(default = 0)
    dx_tran: float = field(default = 0)
    dn: float = field(default = 0)
    de_psi: float = field(default = 0)

@dataclass
class VehicleState(PythonMsg):
    '''
    Complete vehicle state (local, global, and input)
    '''
    vehicle_id: int = field(default=1)

    t: float  = field(default = None)    # time in seconds

    x: Position = field(default=None)  # global position

    v: BodyLinearVelocity = field(default=None)  # body linear velocity
    w: BodyAngularVelocity = field(default=None)  # body angular velocity
    a: BodyLinearAcceleration = field(default=None)  # body linear acceleration
    aa: BodyAngularAcceleration = field(default=None)  # body angular acceleration

    q: OrientationQuaternion = field(default=None)  # global orientation (qr, qi, qj, qk)
    e: OrientationEuler = field(default=None)       # global orientation (phi, theta, psi)

    p: ParametricPose = field(default=None)  # parametric position (s,y, ths)
    pt: ParametricVelocity = field(default=None)  # parametric velocity (ds, dy, dths)

    u: VehicleActuation = field(default=None)

    def __post_init__(self):
        if self.x is None: self.x = Position()
        if self.u is None: self.u = VehicleActuation()
        if self.v is None: self.v = BodyLinearVelocity()
        if self.w is None: self.w = BodyAngularVelocity()
        if self.a is None: self.a = BodyLinearAcceleration()
        if self.aa is None: self.aa = BodyAngularAcceleration()
        if self.q is None: self.q = OrientationQuaternion()
        if self.e is None: self.e = OrientationEuler()
        if self.p is None: self.p = ParametricPose()
        if self.pt is None: self.pt = ParametricVelocity()
        return

    def get_R(self, reverse = False):
        #Warning - Not suitable for general 3D case
        psi = self.psi
        return np.array([[np.cos(psi), -np.sin(psi),0],[np.sin(psi),np.cos(psi),0],[0,0,0]])

    # Single operation with rotation matrix
    # def update_body_velocity_from_global(self):
    #     self.v_long =  self.v_x * np.cos(self.psi) + self.v_y * np.sin(self.psi)
    #     self.v_tran = -self.v_x * np.sin(self.psi) + self.v_y * np.cos(self.psi)
    #     self.a_long =  self.a_x * np.cos(self.psi) + self.a_y * np.sin(self.psi)
    #     self.a_tran = -self.a_y * np.sin(self.psi) + self.a_y + np.cos(self.psi)
    #
    # def update_global_velocity_from_body(self):
    #     self.v_x =  self.v_long * np.cos(self.psi) - self.v_tran * np.sin(self.psi)
    #     self.v_y =  self.v_long * np.sin(self.psi) + self.v_tran * np.cos(self.psi)
    #     self.a_x =  self.a_long * np.cos(self.psi) - self.a_tran * np.sin(self.psi)
    #     self.a_y =  self.a_long * np.sin(self.psi) + self.a_tran * np.cos(self.psi)

    def pack_list(self, use_numpy = False):
        '''
        Takes in a list or array of VehicleState objects and creates a single VehicleState object where each field is a np.array of the fields of the original list

        usage: pytypes.VehicleState.pack_list(array_of_VehicleState)
        '''
        msg = VehicleState()
        N = len(self)
        for key in vars(msg).keys():
            msg.__setattr__(key,[])

            for i in range(N):
                msg.__getattribute__(key).append(self[i].__getattribute__(key))
            if use_numpy:
                msg.__setattr__(key, np.array(msg.__getattribute__(key)))

        return msg

    def copy_control(self, destination):
        '''
        copies control state form self to destination
        '''
        destination.t = self.t
        destination.u_a = self.u.u_a
        destination.u_steer = self.u.u_steer
        return

# TODO: Change to array of VehicleState
@dataclass
class VehiclePrediction(PythonMsg):
    '''
    Complete vehicle coordinates (local, global, and input)
    '''
    t: float  = field(default = None)    # time in seconds

    x: array.array = field(default = None)     # global x coordinate in meters
    y: array.array = field(default = None)     # global y coordinate in meters

    v: array.array = field(default = None)
    v_x: array.array = field(default = None)   # global x velocity in m/s
    v_y: array.array = field(default = None)   # global y velocity in m/s

    a_x: array.array = field(default = None)   # global x acceleration in m/s^2
    a_y: array.array = field(default = None)   # global y acceleration in m/s^2

    psi: array.array = field(default = None)      # global vehicle heading angle
    psidot: array.array = field(default = None)   # global and local angular velocity of car

    v_long: array.array = field(default = None)   # longitudinal velocity (in the direction of psi)
    v_tran: array.array = field(default = None)   # transverse velocity   (orthogonal to the direction of psi)

    a_long: array.array = field(default = None)   # longitudinal velocity (in the direction of psi)
    a_tran: array.array = field(default = None)   # transverse velocity   (orthogonal to the direction of psi)

    e_psi: array.array = field(default = None)    # heading error between car and track
    s: array.array = field(default = None)        # path length along center of track to projected position of car
    x_tran: array.array = field(default = None)   # deviation from centerline (transverse position)

    u_a: array.array = field(default = None)      # acceleration output
    u_steer: array.array = field(default = None)  # steering angle output

    lap_num: int = field(default = None)

    #TODO: Maybe need a different way of storing the covariance matricies
    # For covariances, only store the upper triangular part
    local_state_covariance: array.array = field(default = None) # Vectorized upper triangular part of covariance matrix with main diagonal order [v_long, v_tran, psidot, e_psi, s, e_y]
    global_state_covariance: array.array = field(default = None) # Vectorized upper triangular part of covariance matrix with main diagonal order [x, y, psi, v_long, v_tran, psidot]

    def update_body_velocity_from_global(self):
        self.v_long =  (np.multiply(self.v_x, np.cos(self.psi)) + np.multiply(self.v_y, np.sin(self.psi))).tolist()
        self.v_tran = (-np.multiply(self.v_x, np.sin(self.psi)) + np.multiply(self.v_y, np.cos(self.psi))).tolist()
        self.a_long =  (np.multiply(self.a_x, np.cos(self.psi)) + np.multiply(self.a_y, np.sin(self.psi))).tolist()
        self.a_tran = (-np.multiply(self.a_y, np.sin(self.psi)) + np.multiply(self.a_y, np.cos(self.psi))).tolist()

    def update_global_velocity_from_body(self):
        self.v_x =  (np.multiply(self.v_long, np.cos(self.psi)) - np.multiply(self.v_tran, np.sin(self.psi))).tolist()
        self.v_y =  (np.multiply(self.v_long, np.sin(self.psi)) + np.multiply(self.v_tran, np.cos(self.psi))).tolist()
        self.a_x =  (np.multiply(self.a_long, np.cos(self.psi)) - np.multiply(self.a_tran, np.sin(self.psi))).tolist()
        self.a_y =  (np.multiply(self.a_long, np.sin(self.psi)) + np.multiply(self.a_tran, np.cos(self.psi))).tolist()


