import casadi as ca
from parksim.vehicle_types import VehicleBody


def kinematic_bicycle_ct(vehicle_body: VehicleBody):
    """
    return the continuous time kinematic bicycle model
    """
    x = ca.SX.sym("x")
    y = ca.SX.sym("y")
    v = ca.SX.sym("v")
    yaw = ca.SX.sym("yaw")
    state = ca.vertcat(x, y, yaw, v)

    a = ca.SX.sym("a")
    delta = ca.SX.sym("delta")
    input = ca.vertcat(a, delta)

    xdot = v * ca.cos(yaw)
    ydot = v * ca.sin(yaw)
    vdot = a
    yawdot = v / vehicle_body.wb * ca.tan(delta)
    output = ca.vertcat(xdot, ydot, yawdot, vdot)

    return ca.Function("f_ct", [state, input], [output])


def kinematic_bicycle_rk(dt: float, vehicle_body: VehicleBody, M=4):
    """
    rk discrete time model for the kinematic bicycle dynamics
    """
    h = dt / M

    x = ca.SX.sym("x")
    y = ca.SX.sym("y")
    yaw = ca.SX.sym("yaw")
    v = ca.SX.sym("v")
    state = ca.vertcat(x, y, yaw, v)

    a = ca.SX.sym("a")
    delta = ca.SX.sym("delta")
    input = ca.vertcat(a, delta)

    f_ct = kinematic_bicycle_ct(vehicle_body=vehicle_body)

    zkp = state
    for _ in range(M):
        a1 = f_ct(zkp, input)
        a2 = f_ct(zkp + h * a1 / 2, input)
        a3 = f_ct(zkp + h * a2 / 2, input)
        a4 = f_ct(zkp + h * a3, input)

        zkp = zkp + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)

    return ca.Function("f_dt", [state, input], [zkp])
