from typing import final
import casadi as ca
from casadi.casadi import LABEL
import numpy as np

from typedef.obstacle_types import *
from typedef.pytypes import VehiclePrediction
from typedef.vehicle_types import VehicleBody, VehicleState

from visualizer.offline_visualizer import OfflineVisualizer

from utils.interpolation import interpolate_states_inputs 

@dataclass
class CollocationPlanConfig(PythonMsg):
    K: int = field(default = 8)
    N: int = field(default = 10)
    h0: float = field(default = 0.5)
    v_max: float = field(default = 1)
    uy_max: float = field(default = 0.5)
    ua_max: float = field(default = 0.5)
    
    d_min: float = field(default = .5)
    
    Q: np.ndarray = field(default = np.eye(4)*0)
    R: np.ndarray = field(default = np.eye(2)*10)
    
    def get_collocation_coefficients(self):
        K = self.K
        
        tau = np.append(0, ca.collocation_points(K, 'radau')) # legendre or radau, radau is usually better

        B = np.zeros(K+1)      # collocation coefficients for quadrature (integral)
        C = np.zeros((K+1,K+1))# collocation coefficients for continuity within interval (derivative at each point)
        D = np.zeros(K+1)      # collocation coefficients for continuity at end of interval (last value)

        for j in range(K+1):
            p = np.poly1d([1])
            for r in range(K+1):
              if r != j:
                  p *= np.poly1d([1, -tau[r]]) / (tau[j] - tau[r])

            integral = np.polyint(p)
            B[j] = integral(1.0)

            tangent = np.polyder(p)
            for r in range(K+1):
                C[j,r] = tangent(tau[r])

            D[j] = p(1.0)
        
        return B, C, D
        

class CollocationPlanner():
    def __init__(self, config: CollocationPlanConfig,
                       vehicle_body: VehicleBody,
                       region: GeofenceRegion):
        
        assert isinstance(config, CollocationPlanConfig)
        assert isinstance(vehicle_body, VehicleBody)
        assert isinstance(region, GeofenceRegion)
        
        
        self.config = config
        self.vehicle_body = vehicle_body
        self.region = region
        return
        
    
    def setup(self, obstacles = []):
        self.obstacles = obstacles
        n_obs = len(self.obstacles)
        
        B, C, D = self.config.get_collocation_coefficients()
        self._setup_bicycle_model()
        
        state_u = [np.inf, self.region.x_max, self.region.y_max, np.inf, self.config.v_max]
        state_l = [-np.inf, self.region.x_min, self.region.y_min, -np.inf,-self.config.v_max]
        
        input_u = [self.config.ua_max, self.config.uy_max]
        input_l = [-self.config.ua_max, -self.config.uy_max]
        
        
        veh_G = self.vehicle_body.A
        veh_g = self.vehicle_body.b
        
        p = []       # parameters (initial state, cost terms, terminal terms)
        w = []       # states     (vehicle state and inputs)
        w0 = []      
        lbw = []
        ubw = []
        J = 0        # cost
        g = []       # nonlinear constraint functions
        lbg = []
        ubg = []
        
        # initial and final state parameters
        state_init = ca.SX.sym('z0', self.f_state.size())
        state_final = ca.SX.sym('zf', self.f_state.size())
        p += [state_init]
        p += [state_final]
        
        
        X = np.resize(np.array([], dtype = ca.SX), (self.config.N, self.config.K+1))
        U = np.resize(np.array([], dtype = ca.SX), (self.config.N, self.config.K+1))
        L = np.resize(np.array([], dtype = ca.SX), (self.config.N, self.config.K+1, n_obs))
        M = np.resize(np.array([], dtype = ca.SX), (self.config.N, self.config.K+1, n_obs))

        h = ca.SX.sym('h')
        w += [h]
        w0 += [self.config.h0]
        ubw += [self.config.h0*4]
        lbw += [self.config.h0/4]
        
        # first create state and input variables, ordered in a way that is easy to unpack
        for k in range(self.config.N):
            # add collocation point state and inputvariables
            for j in range(0, self.config.K+1):
                xkj = ca.SX.sym('x_%d_%d'%(k,j), self.f_state.size())
                X[k,j] = xkj
                w += [xkj]
                w0 += [0.] * self.f_state.size()[0]
                lbw += state_l
                ubw += state_u
                
                ukj = ca.SX.sym('u_%d_%d'%(k,j), self.f_input.size())
                U[k,j] = ukj
                w += [ukj]
                w0 += [0.] * self.f_input.size()[0]
                lbw += input_l
                ubw += input_u
                
            
        # add obstacle avoidance dual variables and constraints
        for k in range(self.config.N):   
            for j in range(0, self.config.K+1):   
                yaw = X[k,j][3]
                R = ca.vertcat(ca.horzcat(ca.cos(yaw), - ca.sin(yaw)),
                                   ca.horzcat(ca.sin(yaw),   ca.cos(yaw)))
                t = X[k,j][1:3]
                 
                for m in range(n_obs):
                    if isinstance(self.obstacles[m], RectangleObstacle):
                        # polytope - polytope OBCA with four hyperplanes for the vehicle and obstacle
                        
                        obs_A = obstacles[m].A
                        obs_b = obstacles[m].b
                        
                        # add dual variables
                        lkjm = ca.SX.sym('l_%d_%d_%d'%(k,j,m), 4) 
                        L[k,j,m] = lkjm
                        w += [lkjm]
                        w0 += [0.]*4
                        lbw += [0.]*4
                        ubw += [np.inf]*4
                    
                        mkjm = ca.SX.sym('m_%d_%d_%d'%(k,j,m), 4)
                        M[k,j,m] = mkjm
                        w += [mkjm]
                        w0 += [0.]*4
                        lbw += [0.]*4
                        ubw += [np.inf]*4
                        
                        # add constraints
                        c1 =  ca.dot(-veh_g, M[k,j,m]) + (obs_A @ t - obs_b).T @ L[k,j,m] # greater than d_min
                        c2 =  veh_G.T @ M[k,j,m] + R.T @ obs_A.T @ L[k,j,m]  # = [0,0]
                        c3 = ca.dot(obs_A.T @ L[k,j,m], obs_A.T @ L[k,j,m]) # <= 1
                            
                        g +=  [c1,c2,c3]
                        ubg += [np.inf, 0, 0, 1]
                        lbg += [self.config.d_min, 0, 0, -np.inf]  
                        
                    elif isinstance(self.obstacles[m], CircleObstacle):
                        #polytope - point OBCA with d_min increased by the radius of the circular obstacle
                        
                        # add dual variables
                        lkjm = ca.SX.sym('l_%d_%d_%d'%(k,j,m), 4) 
                        L[k,j,m] = lkjm
                        w += [lkjm]
                        w0 += [0.]*4
                        lbw += [0.]*4
                        ubw += [np.inf]*4
                        
                        # add constraints
                        #TODO - fix for rotation and translation of obstacle rather than vehicle
                        c1 = (ca.dot(veh_G, t) - veh_g).T @ L[k,j,m]        # > d_min + r
                        c2 = ca.dot(veh_G.T @ L[k,j,m], veh_G.T @ L[k,j,m]) # <= 1
                        
                        g += [c1,c2]
                        ubg += [np.inf, 1]
                        lbg += [self.config.d_min + obstacles[m].r, -np.inf]
                        
                        
                        
                    else: 
                        raise NotImplementedError('Unknown Obstacle Class: %s'%type(obstacles[m]).__name__)

                    
        # add general constraints            
        for k in range(self.config.N):   
        
            # add ode continuity constraints within the interval
            for j in range(0, self.config.K+1):
                poly_ode = 0
                for j2 in range(self.config.K+1):
                  poly_ode += C[j2][j] * X[k,j2] / h #H[k,j]
                
                func_ode = self.f(X[k,j],U[k,j])

                g += [func_ode - poly_ode]
                ubg += [0.] * self.f_state.size()[0]
                lbg += [0.] * self.f_state.size()[0]
            
            # add quadrature costs 
            if True:
                poly_int = 0
                for j in range(self.config.K+1):
                    stage_cost = ca.bilin(self.config.Q, X[k,j][1:] - state_final[1:], X[k,j][1:] - state_final[1:]) + ca.bilin(self.config.R, U[k,j], U[k,j])
                    poly_int += B[j] * stage_cost * h
                J += poly_int
          
            # add state continuity constraints
            if k >= 1:
                poly_prev = 0
                for j in range(self.config.K+1):
                    poly_prev += X[k-1,j] * D[j]
                
                g += [X[k,0] - poly_prev]
                ubg += [0.] * self.f_state.size()[0]
                lbg += [0.] * self.f_state.size()[0]
        
        
        # initial state constraint - don't bother with t since we can just subtract the offset, this helps the solver substantially
        #TODO - change this from simply leaving x coordinate unconstrained to making the inital state constrained to lie along a line. 
        g += [X[0,0][1:] - state_init[1:]]
        ubg += [0.] * (self.f_state.size()[0] - 1)
        lbg += [0.] * (self.f_state.size()[0] - 1)
        # g += [X[0,0] - state_init]
        # ubg += [0.] * self.f_state.size()[0]
        # lbg += [0.] * self.f_state.size()[0]
        
        
        # target state constraint
        g += [X[-1,-1][1:] - state_final[1:]]
        ubg += [0.] * (self.f_state.size()[0] - 1)
        lbg += [0.] * (self.f_state.size()[0] - 1)
        
        
        
        
        # penalize final time
        delta_t = X[-1,-1][0] - X[0,0][0]
        J += delta_t     
        
        
               
        prob = {'f':J,'x':ca.vertcat(*w), 'g':ca.vertcat(*g), 'p':ca.vertcat(*p)}
        opts = {'ipopt.print_level': 5, 'ipopt.sb':'yes','print_time':0}

        solver = ca.nlpsol('solver','ipopt',prob, opts)      
        self.solver = solver
        self.solver_w0 = w0
        self.solver_ubw = ubw
        self.solver_lbw = lbw
        self.solver_ubg = ubg
        self.solver_lbg = lbg
        
        return 
    
    def solve(self, init_state: VehicleState, final_state: VehicleState):
        x0 = [0,init_state.x.x, init_state.x.y, init_state.q.to_yaw(), init_state.v.mag()]
        xf = [0,final_state.x.x, final_state.x.y, final_state.q.to_yaw(), final_state.v.mag()]
        
        
        sol = self.solver(x0 = self.solver_w0,
                          ubx = self.solver_ubw,
                          lbx = self.solver_lbw, 
                          ubg = self.solver_ubg,
                          lbg = self.solver_lbg,
                          p = [*x0, *xf])
            
        return sol

    def pack_solution(self, sol):
        """
        pack solution to VehiclePrediction type
        """
        sol_states = np.array(sol['x'][1:])
        
        sep = 5 + 2
        sol_states = sol_states[0:5*(self.config.N * (self.config.K+1)) + 2*self.config.N * (self.config.K+1)]
        sol_t  = sol_states[np.arange(0,len(sol_states),sep)]
        sol_xi = sol_states[np.arange(1,len(sol_states),sep)]
        sol_xj = sol_states[np.arange(2,len(sol_states),sep)]
        sol_th = sol_states[np.arange(3,len(sol_states),sep)]
        sol_v  = sol_states[np.arange(4,len(sol_states),sep)]
        sol_ua = sol_states[np.arange(5,len(sol_states),sep)]
        sol_uy = sol_states[np.arange(6,len(sol_states),sep)]

        return VehiclePrediction(t=sol_t[:,0]-sol_t[0,0], x=sol_xi[:,0], y=sol_xj[:,0], v=sol_v[:,0], psi=sol_th[:,0], u_a=sol_ua[:,0], u_steer=sol_uy[:,0])

    def _setup_bicycle_model(self):
        lr = self.vehicle_body.lr
        lf = self.vehicle_body.lf
    
        t = ca.SX.sym('t')

        xi = ca.SX.sym('xi')
        xj = ca.SX.sym('xj')
        th = ca.SX.sym('th')
        v = ca.SX.sym('v')

        ua = ca.SX.sym('ua')
        uy = ca.SX.sym('uy')

        beta = ca.atan(lr/(lf+lr) * ca.tan(uy))
        v1 = ca.cos(beta) * v
        v2 = ca.sin(beta) * v
        w3 = v * ca.cos(beta) / (lr + lf) * ca.tan(uy)

        t_dot = 1
        xi_dot = v1 * ca.cos(th) - v2 * ca.sin(th)
        xj_dot = v1 * ca.sin(th) + v2 * ca.cos(th)
        th_dot = w3
        v_dot = ua

        state = ca.vertcat(t,xi, xj, th, v)
        input = ca.vertcat(ua, uy)
        ode   = ca.vertcat(t_dot, xi_dot, xj_dot ,th_dot, v_dot)

        f = ca.Function('f',[state,input],[ode])
        
        self.f = f
        self.f_state = state
        self.f_input = input
        return


def test_planner():
    name = 'test_planner'
    config = CollocationPlanConfig()
    vehicle_body = VehicleBody(vehicle_flag=0)
    region = GeofenceRegion(x_max=8, x_min=-8, y_max=11, y_min=-3)
    
    obstacles = [RectangleObstacle(xc = -3.8,   yc = 0, w = 5, h = 5.22),
                 RectangleObstacle(xc = 3.8, yc = 0, w = 5, h = 5.22),
                 RectangleObstacle(xc = 0,   yc = 10, w = 12, h = 0.8)]
    
    planner = CollocationPlanner(config, vehicle_body, region)
    planner.setup(obstacles)
    
    init_state = VehicleState()
    final_state = VehicleState()
    
    init_state.x.x = -5
    init_state.x.y = 6.1
    init_state.q.from_yaw(0)
    
    final_state.x.x = 0
    final_state.x.y = 0
    final_state.q.from_yaw(np.pi/2)
    
    
    sol = planner.solve(init_state, final_state)
    obca_sol = planner.pack_solution(sol)

    # Intepolate evenly for animation
    interval = 40
    total_time = obca_sol.t[-1] - obca_sol.t[0]
    new_t = np.linspace(obca_sol.t[0], obca_sol.t[-1], int(1000*total_time/interval))
    interp_sol = interpolate_states_inputs(obca_sol, new_t)

    save_csv(interp_sol, name)

    vis = OfflineVisualizer(sol=interp_sol, obstacles=obstacles, map=None, vehicle_body=vehicle_body, region=region)
    
    vis.plot_solution(step=75, fig_path=name+'.png')
    vis.animate_solution(gif_path=name+'.gif')

# def generate_data():
#     from itertools import product

#     config = CollocationPlanConfig()
#     vehicle_body = VehicleBody(vehicle_flag=0)
#     region = GeofenceRegion(x_max=8, x_min=-8, y_max=3, y_min=-11)
    
#     obstacles = [RectangleObstacle(xc = -3.8,   yc = 0, w = 5, h = 5.22),
#                  RectangleObstacle(xc = 3.8, yc = 0, w = 5, h = 5.22),
#                  RectangleObstacle(xc = 0,   yc = -10, w = 12, h = 0.8)]
    
#     planner = CollocationPlanner(config, vehicle_body, region)
#     planner.setup(obstacles)

#     start_x = {'A': -5, 'D': 5}
#     start_y = {'W': -7.85, 'S': -4.35}
#     start_psi = {'W': np.pi, 'S': 0}
#     end_psi = {'F': np.pi/2, 'B': -np.pi/2}

#     for lane, position, method in product('WS', 'AD', 'FB'):
#         if lane == 'W' and position == 'A' and method == 'B':
#             continue

#         if lane == 'S' and position == 'D' and method == 'F':
#             continue

#         name = '2' + lane + position + method

#         init_state = VehicleState()
#         final_state = VehicleState()
        
#         init_state.x.x = start_x[position]
#         init_state.x.y = start_y[lane]
#         init_state.q.from_yaw(start_psi[lane])
        
#         final_state.x.x = 0
#         final_state.x.y = 0
#         final_state.q.from_yaw(end_psi[method])
        
        
#         sol = planner.solve(init_state, final_state)
#         obca_sol = planner.pack_solution(sol)

#         # Intepolate evenly for animation
#         interval = 40
#         total_time = obca_sol.t[-1] - obca_sol.t[0]
#         new_t = np.linspace(obca_sol.t[0], obca_sol.t[-1], int(1000*total_time/interval))
#         interp_sol = interpolate_states_inputs(obca_sol, new_t)

#         save_csv(interp_sol, name)

#         vis = OfflineVisualizer(sol=interp_sol, obstacles=obstacles, map=None, vehicle_body=vehicle_body, region=region)
        
#         vis.plot_solution(step=75, fig_path=name+'.png', show=False)
#         vis.animate_solution(gif_path=name+'.gif', show=False)

def save_csv(interp_sol:VehiclePrediction, name):
    np.savetxt(name+'.csv', np.array([interp_sol.t, interp_sol.x, interp_sol.y, interp_sol.psi, interp_sol.v, interp_sol.u_a, interp_sol.u_steer]).T, delimiter=',', header='t,x,y,heading,v,u_a,u_steer')
    
    return


    
if __name__ == '__main__':
    test_planner()
    # generate_data()
