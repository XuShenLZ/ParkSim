import numpy as np
import pdb
from abc import abstractmethod
from dataclasses import dataclass, field

from mpclab_common.pytypes import PythonMsg

@dataclass
class BaseObstacle(PythonMsg):
    '''
    Base class for 2D obstacles
    
    xy is an N+1 x 2 numpy array corresponding to the N vertices of the obstacle with the first vertex repeated at the end
    
    this is used for plotting the obstacle as a matplotlib Polygon patch.
    '''
    xy: float = field(default = None)

    
    
@dataclass
class BaseConvexObstacle(BaseObstacle):
    '''
    Base class for 2D convex obstacles, intended for use with representation of an obstacle as 2D hyperplanes.
    
    R = conv(V)               (Vertex Form)
    R = {x : Ab*x <= b}       (Hyperplane form)
    R = {x : l <= A*x <= u}   (OSQP)
    '''
    
    V: float = field(default = None)   # vertices - N x 2 shape np.array
    Ab: float = field(default = None)  # single ended constraints
    b: float = field(default = None)
    l: float = field(default = None)
    A: float = field(default = None)
    u: float = field(default = None)
    def __setattr__(self,key,value):
        '''
        Reuse PythonMsg checks however update internal representations afterwards. 
        
        Internal updates must use object.__setattr__(self,key,value) to avoid recursive infinite loops.
        '''
        PythonMsg.__setattr__(self,key,value)
        
        # now update V,Ab,b,l,A,u
        #self.__calc_V__()
        #self.__calc_Ab_b__()
        #self.__calc_l_A_u__()
        return
        
    @abstractmethod
    def __calc_V__(self):
        '''
        Compute vertex form of obstacle in the form of an N x 2 numpy array
        
        sets self.V to the vertices of the obstacles and 
             self.xy to the vertices of the obstacle with the first point repeated at the end (xy is for plotting)
        '''
        return
        
    @abstractmethod
    def __calc_Ab_b__(self):
        '''
        Compute single-sided hyperplane form of the polytope:
        
        P = {x : Ab @ x <= b}
        '''
        return
        
    @abstractmethod
    def __calc_l_A_u__(self):
        '''
        Compute single-sided hyperplane form of the polytope:
        
        P = {x : l <= A @ x <= u}
        '''
        return



@dataclass
class RectangleObstacle(BaseConvexObstacle):
    '''
    Stores a rectangle and computes polytope representations from it
    
    it is intended that xc,yc,w,h,psi are changed. The rest are computed from these fields. 
    '''
    xc: float = field(default = 0)
    yc: float = field(default = 0)
    w: float = field(default = 0)
    h: float = field(default = 0)
    psi: float = field(default = 0)
    
    
    def __post_init__(self):
        self.__calc_V__()
        self.__calc_Ab_b__()
        self.__calc_l_A_u__()
        return
        
    def R(self):
        return np.array([[np.cos(self.psi), np.sin(self.psi)],[-np.sin(self.psi), np.cos(self.psi)]]) 
    
    def __calc_V__(self):
        xy = np.array([[- self.w/2,- self.h/2], [- self.w/2,+ self.h/2], [+ self.w/2,+ self.h/2], [+ self.w/2,- self.h/2], [- self.w/2,- self.h/2]]) @ self.R() + np.array([[self.xc, self.yc]])
        
        V = xy[:-1,:]
                           
        object.__setattr__(self,'xy',xy)
        object.__setattr__(self,'V',V  )
        return
        
    def __calc_Ab_b__(self):
        Ab = np.array([[1,0],
                            [0,1],
                            [-1,0],
                            [0,-1]])  @ self.R()
                            
        A = np.eye(2) @ self.R()
        l = (np.linalg.inv(A.T) @ np.array([[-self.xc],[-self.yc]])).squeeze() + np.array([self.w/2, self.h/2])  
        u = (np.linalg.inv(A.T) @ np.array([[ self.xc],[ self.yc]])).squeeze() + np.array([self.w/2, self.h/2])    
        b = np.concatenate([u,l])
        
        object.__setattr__(self,'Ab',Ab)
        object.__setattr__(self,'b',b)
        return
    
    def __calc_l_A_u__(self):
        A = np.eye(2) @ self.R()
        l = (np.linalg.inv(A.T) @ np.array([[self.xc],[self.yc]])).squeeze() - np.array([self.w/2, self.h/2])
        u = (np.linalg.inv(A.T) @ np.array([[self.xc],[self.yc]])).squeeze() + np.array([self.w/2, self.h/2])
        
        object.__setattr__(self,'A',A)
        object.__setattr__(self,'l',l)
        object.__setattr__(self,'u',u)
        return
    
    def circumscribed_ellipse(self):
        '''
        places a rotated ellipse at the rectangle's position with identical aspect ratio that contains it entirely
        
        this should really have a type of its own
        '''
        xc = self.xc
        yc = self.yc
        psi = self.psi
        
        a = self.w/2
        b = self.h/2
        
        return xc,yc,a,b,psi
        
    
    def circumscribed_circle(self):
        '''
        places a circle at the rectangle's position that contains it in entirety
        
        this shouldreally have a type of its own for returning
        '''
        xc = self.xc
        yc = self.yc
        R  = np.sqrt(self.w **2 + self.h **2) / 4
        return xc,yc,R
        

def demo_rectangle_obstacle():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    
    r = RectangleObstacle(xc = 5, yc = 10, w = 4, h = 2, psi = 0.4)
    patch = Polygon(r.xy)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    print(r.xy)
    print(r.A)
    print(r.l)
    print(r.u)
    ax.add_patch(patch)
    ax.relim()
    ax.autoscale_view()
    plt.show()
    
    return


def test_rectangle_obstacle():
    itr = 1000
    eps = 1e-9
    for j in range(itr):
        xc = np.random.uniform(-10,10)
        yc = np.random.uniform(-10,10)
        w = np.random.uniform(1,10)
        h = np.random.uniform(1,10)
        psi = np.random.uniform(0,10)
        
        r = RectangleObstacle(xc = xc, yc = yc, w = w, h = h, psi = psi)
        for vertex in range(4):
            assert     np.all(r.Ab @ r.xy[vertex, :] <= r.b + eps)
            assert not np.all(r.Ab @ r.xy[vertex, :] <= r.b - eps)
            
            assert     np.all(r.A  @ r.xy[vertex, :] <= r.u + eps)
            assert     np.all(r.A  @ r.xy[vertex, :] >= r.l - eps)
            assert not np.all(r.A  @ r.xy[vertex, :] <= r.u - eps) or not np.all(r.A  @ r.xy[vertex, :] >= r.l + eps)
    return  

def main():
    demo_rectangle_obstacle()
    #test_rectangle_obstacle()
    return

if __name__ == '__main__':
    main()
    
    
    
    
