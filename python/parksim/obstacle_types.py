import numpy as np
from abc import abstractmethod
from dataclasses import dataclass, field
from matplotlib.patches import Polygon, Circle

from parksim.pytypes import PythonMsg


@dataclass
class GeofenceRegion:
    x_max: float = field(default=140)
    x_min: float = field(default=0)
    y_max: float = field(default=80)
    y_min: float = field(default=0)

    def xy(self):
        return np.array(
            [
                [self.x_max, self.y_max],
                [self.x_max, self.y_min],
                [self.x_min, self.y_min],
                [self.x_min, self.y_max],
                [self.x_max, self.y_max],
            ]
        )


@dataclass
class BaseObstacle(PythonMsg):
    """
    Base class for 2D obstacles

    xy is an N+1 x 2 numpy array corresponding to the N vertices of the obstacle with the first vertex repeated at the end

    this is used for plotting the obstacle as a matplotlib Polygon patch.
    """

    xy: np.ndarray = field(default=None)


@dataclass
class CircleObstacle(BaseObstacle):
    """
    a circular obstacle; the field "xy" should not be used, most software has tools for plotting nice circles
    """

    xc: float = field(default=None)
    yc: float = field(default=None)
    r: float = field(default=None)

    def plot_pyplot(self, ax):
        p = Circle((self.xc, self.yc), radius=self.r, color="red")
        ax.add_patch(p)
        return


@dataclass
class BasePolytopeObstacle(BaseObstacle):
    """
    Base class for 2D convex obstacles, intended for use with representation of an obstacle as 2D hyperplanes.

    R = conv(V)               (Vertex Form)
    R = {x : Ab*x <= b}       (Hyperplane form)
    R = {x : l <= A*x <= u}   (OSQP)
    """

    V: np.ndarray = field(default=None)  # vertices - N x 2 shape np.array
    A: np.ndarray = field(default=None)  # single ended constraints
    b: np.ndarray = field(default=None)

    def __setattr__(self, key, value):
        """
        Reuse PythonMsg checks however update internal representations afterwards.

        Internal updates must use object.__setattr__(self,key,value) to avoid recursive infinite loops.
        """
        PythonMsg.__setattr__(self, key, value)

        # now update dependent fields
        self.__calc_V__()
        self.__calc_A_b__()
        return

    @abstractmethod
    def __calc_V__(self):
        """
        Compute vertex form of obstacle in the form of an N x 2 numpy array

        sets self.V to the vertices of the obstacles and
             self.xy to the vertices of the obstacle with the first point repeated at the end (xy is for plotting)
        """
        return

    @abstractmethod
    def __calc_A_b__(self):
        """
        Compute single-sided hyperplane form of the polytope:

        P = {x : A @ x <= b}
        """
        return

    def plot_pyplot(self, ax):
        p = Polygon(self.xy, color="#7f7f7f")
        ax.add_patch(p)
        return


@dataclass
class RectangleObstacle(BasePolytopeObstacle):
    """
    Stores a rectangle and computes polytope representations from it

    it is intended that xc,yc,w,h,psi are changed. The rest are computed from these fields and are updated whenever a field is set.

    This is not suitable for a vehicle, where it is important to reference the rear axle, see vehicle_types.py
    """

    xc: float = field(default=0)
    yc: float = field(default=0)
    w: float = field(default=0)
    h: float = field(default=0)
    psi: float = field(default=0)

    def __post_init__(self):
        self.__calc_V__()
        self.__calc_A_b__()
        return

    def R(self):
        return np.array(
            [
                [np.cos(self.psi), np.sin(self.psi)],
                [-np.sin(self.psi), np.cos(self.psi)],
            ]
        )

    def __calc_V__(self):
        xy = np.array(
            [
                [-self.w / 2, -self.h / 2],
                [-self.w / 2, +self.h / 2],
                [+self.w / 2, +self.h / 2],
                [+self.w / 2, -self.h / 2],
                [-self.w / 2, -self.h / 2],
            ]
        ) @ self.R() + np.array([[self.xc, self.yc]])

        V = xy[:-1, :]

        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "V", V)
        return

    def __calc_A_b__(self):
        A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]) @ self.R()

        # TODO: there is probably a more efficient version of this.
        R = self.R()
        l = np.linalg.solve(
            R.T, np.array([[-self.xc], [-self.yc]])
        ).squeeze() + np.array([self.w / 2, self.h / 2])
        u = np.linalg.solve(R.T, np.array([[self.xc], [self.yc]])).squeeze() + np.array(
            [self.w / 2, self.h / 2]
        )
        b = np.concatenate([u, l])

        object.__setattr__(self, "A", A)
        object.__setattr__(self, "b", b)
        return


def demo_rectangle_obstacle():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    r = RectangleObstacle(xc=5, yc=10, w=4, h=2, psi=0.4)
    r.xc = 8
    # patch = Polygon(r.xy)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    r.plot_pyplot(ax)
    # print(r.xy)
    # print(r.A)
    # print(r.b)
    # ax.add_patch(patch)

    ax.relim()
    ax.autoscale_view()
    plt.show()

    return


def test_rectangle_obstacle():
    itr = 1000
    eps = 1e-9
    for j in range(itr):
        xc = np.random.uniform(-10, 10)
        yc = np.random.uniform(-10, 10)
        w = np.random.uniform(1, 10)
        h = np.random.uniform(1, 10)
        psi = np.random.uniform(0, 10)

        r = RectangleObstacle(xc=xc, yc=yc, w=w, h=h, psi=psi)
        for vertex in range(4):
            assert np.all(r.A @ r.xy[vertex, :] <= r.b + eps)
            assert not np.all(r.A @ r.xy[vertex, :] <= r.b - eps)

    return


def main():
    demo_rectangle_obstacle()
    test_rectangle_obstacle()
    return


if __name__ == "__main__":
    main()
