#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
import array
import yaml

from typing import List, Dict

SHOW_MSG_TRANSFER_WARNINGS = False

from parksim.pytypes import PythonMsg


class MPClabNode(Node):

    def __init__(self, label: str):
        super().__init__(label)
        return

    def get_ros_time(self):
        time = self.get_clock().now()
        return time.nanoseconds / 1000000000

    def is_valid_parameter_type(self, value):
        '''
        Method for checking if a variable "value" is a valid default value for a ROS2 parameter
        should not be used dynamically for transferring data to/fram ROS2 messages
        adapted from https://github.com/ros2/rclpy/blob/master/rclpy/rclpy/parameter.py
        method "from_parameter_value"
        Will discard numpy types
        current workaround is "is_valid_numpy_parameter_type"
        Has to be handled separately so that the default value can be converted to a list.
        '''
        if value is None:
            return True
        elif isinstance(value, (bool, int, float, str)):
            return True
        elif isinstance(value, (list, tuple, array.array)):
            if all(isinstance(v, bytes) for v in value):
                return True
            elif all(isinstance(v, bool) for v in value):
                return True
            elif all(isinstance(v, int) for v in value):
                return True
            elif all(isinstance(v, float) for v in value):
                return True
            elif all(isinstance(v, str) for v in value):
                return True
        return False

    def is_valid_numpy_parameter_type(self, value):
        '''
        Numpy types are not natively supported by ROS so they have to be converted to lists when declared as default parameter values
        this function is used so that PythonMsg classes can have numpy default values and be used as templates to autodeclare_parameters
        THIS DOES NOT keep track of such parameters to convert autoloaded configurations to numpy - all array-type parameters are loaded as lists
        currently code to do that does not exist, but may exist at a later date.
        '''
        if isinstance(value, np.ndarray):
            if np.issubdtype(value.dtype, np.number):
                return True
        return False

    def autodeclare_parameters(self, template, namespace, verbose=False):
        '''
        autodeclares parameters for the class
        template should be a class with fields corresponding to the parameters needed, e.g. template.dt, template.n_laps
        adds nested parameters for any instance of PythonMsg found in template
        '''

        def unpack_pythonmsg(msg, prefix):
            msg_parameters = []
            for key in vars(msg):
                val = msg.__getattribute__(key)
                if isinstance(val, PythonMsg):
                    nested_parameters = unpack_pythonmsg(val, prefix + '.' + key)
                    for param in nested_parameters:
                        msg_parameters.append(param)
                if self.is_valid_parameter_type(val):
                    msg_parameters.append((prefix + '.' + key, val))
                elif self.is_valid_numpy_parameter_type(val):
                    msg_parameters.append((prefix + '.' + key, val.tolist()))

            return msg_parameters

        parameters = []
        for key in vars(template):
            if verbose:
                self.get_logger().info('Checking parameter: %s' % str(key))
            val = template.__getattribute__(key)
            if isinstance(val, PythonMsg):
                msg_parameters = unpack_pythonmsg(val, key)
                for param in msg_parameters:
                    parameters.append(param)
                    if verbose:
                        self.get_logger().info('Adding parameter: %s' % str(param))
            else:
                parameters.append((key, val))
                if verbose:
                    self.get_logger().info('Adding parameter: %s' % str((key, val)))

        self.declare_parameters(
            namespace=namespace,
            parameters=parameters
        )
        return

    def autoload_parameters(self, template, namespace, suppress_warnings = False, verbose = False):
        '''
        after parameters have been declared, this attempts to load them using the same template and namespace
        rather than modifying the template, loaded fields are added to self
        '''
        for key in vars(template):
            if isinstance(template.__getattribute__(key), PythonMsg):
                msg = template.__getattribute__(key).copy()
                self.unpack_config_parameters(namespace + '.' + key, msg, suppress_warnings = suppress_warnings, verbose = verbose)
                object.__setattr__(self, key, msg)
            else:
                param = '.'.join((namespace,key))
                try:
                    loaded_val = self.get_parameter(param).value
                except rclpy.exceptions.ParameterNotDeclaredException:
                    loaded_val = template.__getattribute__(key)
                    if not suppress_warnings:
                        self.get_logger().warn('Unable to load node param: %s' %(namespace+key))
                object.__setattr__(self, key, loaded_val)
                if verbose:
                    self.get_logger().info('Loaded parameter %s with value %s'%(str(param), str(loaded_val)))
        return

    def load_parameter(self, parameter_name):
        '''
        Loads a single parameter
        '''
        self.declare_parameters(namespace=self.get_namespace(), parameters=[(parameter_name,)])
        name = '.'.join((self.get_namespace(), parameter_name))
        param = self.get_parameter(name)
        self.undeclare_parameter(name)
        return param.value

    def unpack_config_parameters(self, namespace, target_python_msg, suppress_warnings=False, verbose=False):
        '''
            Takes a namespace and tries to find keys in it to load into
            'target_python_msg'
            ex:  vehicle_model = self.load_parameters('vehicle_config/', VehicleConfig())
        '''

        if not namespace == '' and not namespace[-1] == '.': namespace += '.'

        for key in vars(target_python_msg):
            target_data = target_python_msg.__getattribute__(key)

            if isinstance(target_data, PythonMsg):
                self.unpack_config_parameters(namespace + key, target_data, suppress_warnings=suppress_warnings,
                                              verbose=verbose)

            if self.is_valid_parameter_type(target_data):
                try:
                    param = namespace + key
                    val = self.get_parameter(param).value
                    target_python_msg.__setattr__(key, val)
                    if verbose:
                        self.get_logger().info('Loaded parameter %s with value %s' % (param, val))

                except rclpy.exceptions.ParameterNotDeclaredException:
                    if not suppress_warnings:
                        self.get_logger().warn('Unable to load node param: %s' % (namespace + key))
            elif self.is_valid_numpy_parameter_type(target_data):
                try:
                    param = namespace + key
                    val = self.get_parameter(param).value
                    target_python_msg.__setattr__(key, np.array(val))
                    if verbose:
                        self.get_logger().info('Loaded parameter %s with value %s' % (param, val))

                except rclpy.exceptions.ParameterNotDeclaredException:
                    if not suppress_warnings:
                        self.get_logger().warn('Unable to load node param: %s' % (namespace + key))

        return target_python_msg

    def populate_msg(self, msg, data):
        '''
        Takes a python message 'data' and tries to load all of its keys data into
        the ROS2 message 'msg'
        Prints a warning if there is no destination for the key or in the event of type mismatch.
        '''
        for key in vars(data):
            if hasattr(msg, key):
                if isinstance(data.__getattribute__(key), PythonMsg):
                    MPClabNode.populate_msg(self, msg.__getattribute__(key), data.__getattribute__(key))
                    continue
                try:
                    new_data = data.__getattribute__(key)
                    if new_data is None:
                        continue
                    target_type = type(msg.__getattribute__(key))
                    if type(new_data) == target_type:
                        msg.__setattr__(key, new_data)
                    else:
                        converter_str = '__' + target_type.__name__ + '__'
                        # already in try-except so just try getting the converter
                        # hasattr uses a try-except so using that would just be slower
                        converter = getattr(new_data, converter_str)
                        if callable(converter):
                            converted_data = converter()
                            msg.__setattr__(key, converted_data)
                except AssertionError:
                    err = str('Type error for key %s, cannot write type %s to %s' % (
                    key, str(type(data.__getattribute__(key))), str(type(msg.__getattribute__(key)))))
                    if self:
                        self.get_logger().warn(err)
                    else:
                        print(err)
            elif SHOW_MSG_TRANSFER_WARNINGS:
                err = 'No destination for key %s in msg %s' % (key, type(msg))
                if self:
                    self.get_logger().warn(err)
                else:
                    print(err)

        return msg

    def unpack_msg(self, msg, data):
        '''
        Takes a ROS2 message 'msg' and tries to load all of its data into
        the python message 'data'
        Prints a warning if there is no destination for the key
        '''
        for key in msg.get_fields_and_field_types().keys():
            if key == 'header':
                continue
            elif not hasattr(data, key):
                if SHOW_MSG_TRANSFER_WARNINGS:
                    err = 'No destination for key %s in data %s' % (key, type(data))
                    if self:
                        self.get_logger().warn(err)
                    else:
                        print(err)
            elif isinstance(data.__getattribute__(key), PythonMsg):
                MPClabNode.unpack_msg(self, msg.__getattribute__(key), data.__getattribute__(key))
                continue
            else:
                data.__setattr__(key, msg.__getattribute__(key))
        return

def read_yaml_file(filename):
    params = []
    with open(filename, 'r') as f:
        p = yaml.load(f, Loader=yaml.FullLoader)
        # Convert to list of dicts
        for (k, v) in p.items():
            params.append({k : v})
    return params
