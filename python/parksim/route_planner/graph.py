import numpy as np
import matplotlib.pyplot as plt

from dlp.visualizer import Visualizer

class Vertex(object):
    """
    Vertex class
    """
    def __init__(self, coords: np.ndarray):
        """
        instantiate the vertex
        coords: (x, y) coordinate, numpy array
        """
        self.coords = coords
        self.children = []
        self.edges = []

    def dist(self, v: 'Vertex'):
        """
        compute the distance between the current vertex and the input v
        """
        return np.linalg.norm(self.coords - v.coords)

    def add_child(self, v: 'Vertex', e: 'Edge'):
        """
        add a vertex as its child
        v: the vertex to be added
        e: the connecting edge
        """
        self.children.append(v)
        self.edges.append(e)

    def get_children(self):
        """
        get all children and connecting edges of this vertex
        """
        return self.children, self.edges
    
    def __eq__(self, v: 'Vertex'):
        """
        Eq Method
        """
        return np.array_equal(self.coords, v.coords)
    
    def __hash__(self):
        return id(self)

class Edge(object):
    """
    Edge class
    """
    def __init__(self, v1: 'Vertex', v2: 'Vertex', c):
        """
        Instantiate the edge
        v1, v2: vertices
        c: cost (distance)
        """
        self.v1 = v1
        self.v2 = v2
        self.c = c

class WaypointsGraph(object):
    """
    The connectivity graph of waypoints in the parking lot
    """
    def __init__(self):
        """
        Instantiate the graph
        """
        self.vertices = []
        self.edges = []

    def search(self, coords: np.ndarray):
        """
        search a waypoint that is the closest to the the given coordinates
        """
        min_dist = np.inf
        min_idx = None

        for idx, v in enumerate(self.vertices):
            dist = np.linalg.norm(np.array(coords)[:2] - v.coords)
            if dist < min_dist:
                min_idx = idx
                min_dist = dist

        return min_idx

    def add_waypoint_list(self, waypoint_list: np.ndarray):
        """
        add a list of waypoints into the graph. Connect them in order
        waypoint_list: a Nx2 numpy array of waypoint coordinates
        """
        v_list = []
        for waypoint in waypoint_list:
            v_list.append(Vertex(waypoint))

        num = len(v_list)
        e_list = []

        for i in range(num-1):
            # Calculate the distance to the next waypoint
            dist = v_list[i].dist(v_list[i+1])

            # Forward edge
            edge_fwd = Edge(v1=v_list[i], v2=v_list[i+1], c=dist)
            v_list[i].add_child(v=v_list[i+1], e=edge_fwd)
            e_list.append(edge_fwd)

            # Reverse edge
            edge_rev = Edge(v1=v_list[i+1], v2=v_list[i], c=dist)
            v_list[i+1].add_child(v=v_list[i], e=edge_rev)
            e_list.append(edge_rev)

        # Add the list of vertices and edges to the graph
        self.vertices.extend(v_list)
        self.edges.extend(e_list)
            
    def connect(self, coords_v1: np.ndarray, coords_v2: np.ndarray):
        """
        connect two vertices based on the given two coordinates
        """
        # Search the corresponding vertices
        idx_v1 = self.search(coords_v1)
        idx_v2 = self.search(coords_v2)

        if idx_v1 == idx_v2:
            print("The specified locations are too close to each other. No new connection will be created.")
            return
        
        # Calculate the distance between two waypoints
        dist = self.vertices[idx_v1].dist(self.vertices[idx_v2])

        # Forward edge
        edge_fwd = Edge(v1=self.vertices[idx_v1], v2=self.vertices[idx_v2], c=dist)
        self.vertices[idx_v1].add_child(v=self.vertices[idx_v2], e=edge_fwd)
        self.edges.append(edge_fwd)

        # Reverse edge
        edge_rev = Edge(v1=self.vertices[idx_v2], v2=self.vertices[idx_v1], c=dist)
        self.vertices[idx_v2].add_child(v=self.vertices[idx_v1], e=edge_rev)
        self.edges.append(edge_rev)

    def plot(self, ax = None, plt_ops = {}):
        """
        plot the graph edges
        """

        if ax is None:
            _, ax = plt.subplots()

        for e in self.edges:
            x = [e.v1.coords[0], e.v2.coords[0]]
            y = [e.v1.coords[1], e.v2.coords[1]]
            ax.plot(x, y, **plt_ops)

        return ax

    def setup_with_vis(self, vis: 'Visualizer'):
        """
        set up the graph with the dlp Visualizer object
        """
        for _, coords in vis.waypoints.items():
            self.add_waypoint_list(coords)

        # Connect sections
        self.connect(vis.waypoints['C2'][3], vis.waypoints['R2L'][0])
        self.connect(vis.waypoints['C2'][7], vis.waypoints['R3L'][0])

        self.connect(vis.waypoints['C2'][3], vis.waypoints['R2R'][-1])
        self.connect(vis.waypoints['C2'][7], vis.waypoints['R3R'][-1])

        self.connect(vis.waypoints['R1L'][0], vis.waypoints['R1R'][-1])
        
        # Connect corners
        self.connect(vis.waypoints['C1'][0], vis.waypoints['R1C1BR'][0])
        self.connect(vis.waypoints['R1C1BR'][-1], vis.waypoints['R1L'][-1])
        self.connect(vis.waypoints['C1'][2], vis.waypoints['R2C1TR'][0])
        self.connect(vis.waypoints['R2C1TR'][-1], vis.waypoints['R2L'][-1])
        self.connect(vis.waypoints['C1'][4], vis.waypoints['R2C1BR'][0])
        self.connect(vis.waypoints['R2C1BR'][-1], vis.waypoints['R2L'][-1])
        self.connect(vis.waypoints['C1'][6], vis.waypoints['R3C1TR'][0])
        self.connect(vis.waypoints['R3C1TR'][-1], vis.waypoints['R3L'][-1])
        self.connect(vis.waypoints['C1'][8], vis.waypoints['R3C1BR'][0])
        self.connect(vis.waypoints['R3C1BR'][-1], vis.waypoints['R3L'][-1])
        self.connect(vis.waypoints['C1'][10], vis.waypoints['R4C1TR'][0])
        self.connect(vis.waypoints['R4C1TR'][-1], vis.waypoints['R4L'][-1])
        self.connect(vis.waypoints['C1'][12], vis.waypoints['R4C1BR'][0])
        self.connect(vis.waypoints['R4C1BR'][-1], vis.waypoints['R4L'][-1])
        
        self.connect(vis.waypoints['C2'][0], vis.waypoints['R1C2BL'][0])
        self.connect(vis.waypoints['R1C2BL'][-1], vis.waypoints['R1L'][6])
        self.connect(vis.waypoints['C2'][2], vis.waypoints['R2C2TL'][0])
        self.connect(vis.waypoints['R2C2TL'][-1], vis.waypoints['R2L'][0])
        self.connect(vis.waypoints['C2'][4], vis.waypoints['R2C2BL'][0])
        self.connect(vis.waypoints['R2C2BL'][-1], vis.waypoints['R2L'][0])
        self.connect(vis.waypoints['C2'][6], vis.waypoints['R3C2TL'][0])
        self.connect(vis.waypoints['R3C2TL'][-1], vis.waypoints['R3L'][0])
        self.connect(vis.waypoints['C2'][8], vis.waypoints['R3C2BL'][0])
        self.connect(vis.waypoints['R3C2BL'][-1], vis.waypoints['R3L'][0])
        self.connect(vis.waypoints['C2'][10], vis.waypoints['R4C2TL'][0])
        self.connect(vis.waypoints['R4C2TL'][-1], vis.waypoints['R4L'][0])
        self.connect(vis.waypoints['C2'][12], vis.waypoints['R4C2BL'][0])
        self.connect(vis.waypoints['R4C2BL'][-1], vis.waypoints['R4L'][0])
        
        self.connect(vis.waypoints['C2'][0], vis.waypoints['R1C2BR'][0])
        self.connect(vis.waypoints['R1C2BR'][-1], vis.waypoints['R1R'][-1])
        self.connect(vis.waypoints['C2'][2], vis.waypoints['R2C2TR'][0])
        self.connect(vis.waypoints['R2C2TR'][-1], vis.waypoints['R2R'][-1])
        self.connect(vis.waypoints['C2'][4], vis.waypoints['R2C2BR'][0])
        self.connect(vis.waypoints['R2C2BR'][-1], vis.waypoints['R2R'][-1])
        self.connect(vis.waypoints['C2'][6], vis.waypoints['R3C2TR'][0])
        self.connect(vis.waypoints['R3C2TR'][-1], vis.waypoints['R3R'][-1])
        self.connect(vis.waypoints['C2'][8], vis.waypoints['R3C2BR'][0])
        self.connect(vis.waypoints['R3C2BR'][-1], vis.waypoints['R3R'][-1])
        self.connect(vis.waypoints['C2'][10], vis.waypoints['R4C2TR'][0])
        self.connect(vis.waypoints['R4C2TR'][-1], vis.waypoints['R4R'][-1])
        self.connect(vis.waypoints['C2'][12], vis.waypoints['R4C2BR'][0])
        self.connect(vis.waypoints['R4C2BR'][-1], vis.waypoints['R4R'][-1])

        self.connect(vis.waypoints['R4L'][0], vis.waypoints['R4R'][-1])

        self.connect(vis.waypoints['EXT'][1], vis.waypoints['EXTL'][0])
        self.connect(vis.waypoints['EXT'][1], vis.waypoints['EXTR'][0])
        self.connect(vis.waypoints['EXTL'][0], vis.waypoints['R1L'][-1])
        self.connect(vis.waypoints['EXTR'][0], vis.waypoints['R1L'][-13])

    def _dist_to_edge(self, target_coords: np.ndarray, edge: 'Edge'):
        """
        calculate the distance from a target point to the specific edge
        """
        v1_coords = edge.v1.coords
        v2_coords = edge.v2.coords

        if (target_coords - v2_coords) @ (v2_coords - v1_coords) > 0:
            # If target point is closer to point v2
            dist = np.linalg.norm(target_coords - v2_coords)
        elif (target_coords - v1_coords) @ (v2_coords - v1_coords) < 0:
            # If target point is closer to point v1
            dist = np.linalg.norm(target_coords - v1_coords)
        else:
            x2_x1, y2_y1 = v2_coords - v1_coords
            x1_x0, y1_y0 = v1_coords - target_coords
            dist = abs(x2_x1*y1_y0 - x1_x0*y2_y1) / np.linalg.norm(v2_coords - v1_coords)

        return dist

    def dist_to_graph(self, target_coords: np.ndarray):
        """
        calculate the minimal distance from a traget point to the entire graph
        """
        min_dist = np.inf
        for e in self.edges:
            dist = self._dist_to_edge(target_coords, e)
            if dist < min_dist:
                min_dist = dist

        return min_dist