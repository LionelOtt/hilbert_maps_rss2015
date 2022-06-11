"""Module representing a simple counting occupancy grid map."""


import math
import matplotlib.pyplot as plt
import numpy as np

from util import bounding_box, bresenham, normalize_angle, data_generator


class OccupancyGridmap(object):

    def __init__(self, x_limits, y_limits, resolution):
        """Creates a new OccupancyGrid instance.

        :params x_limits x axis limits
        :params y_limits y axis limits
        :params resolution grid size resolution
        """
        self.x_limits = x_limits
        self.y_limits = y_limits
        #self.x_limits = y_limits
        #self.y_limits = x_limits
        self.resolution = resolution

        x_count = int(math.ceil((x_limits[1] - x_limits[0]) / resolution))
        y_count = int(math.ceil((y_limits[1] - y_limits[0]) / resolution))

        #print(x_count, y_count, x_limits, y_limits)
        self.free = np.zeros((x_count, y_count), dtype=np.uint16)
        self.hit = np.zeros((x_count, y_count), dtype=np.uint16)
        self.occupancy = np.zeros((x_count, y_count))
        self.occupancy.fill(0.5)

    def add(self, pose, scan, max_range=40):
        """Adds a new observation to the grid map.

        :params pose the pose the observation was made from
        :params scan the laser scan ranges of the observation
        """
        start_point = (pose[0], pose[1])
        angle_increment = math.pi / len(scan)
        for i, dist in enumerate(scan):
            # Ignore max range readings
            if dist > max_range:
                continue

            angle = normalize_angle(
                    pose[2] - math.pi + i * angle_increment + (math.pi / 2.0)
            )

            # Add laser endpoint
            end_point = (
                pose[0] + dist * math.cos(angle),
                pose[1] + dist * math.sin(angle)
            )

            self.mark_along_line(start_point, end_point)

    def to_grid(self, coord):
        """Returns the tile key corresponding to the given word coordinates.

        :param x coordinate along the x axis
        :param y coordinate along the y axis
        :return tuple of the corresponding tile indices
        """
        rel_x = coord[0] - self.x_limits[0]
        rel_y = coord[1] - self.y_limits[0]

        if not (0 <= rel_x <= (self.x_limits[1] - self.x_limits[0])) or \
           not (0 <= rel_y <= (self.y_limits[1] - self.y_limits[0])):
            print("Invalid coordinate requested")
            return (0, 0)

        return (
                int(math.floor(rel_x / self.resolution)),
                int(math.floor(rel_y / self.resolution))
        )

    def mark_along_line(self, start_point, end_point):
        """Marks all points along the line between start and end point.

        :params start_point starting location for the laser beam
        :params end_point end location for the laser beam
        """

        # Transform world coordinates to grid coordinates
        grid_start = self.to_grid(start_point)
        grid_end = self.to_grid(end_point)

        # Mark all points on the line between start and end
        coords = bresenham(grid_start, grid_end)
        for pt in coords[:-1]:
            self.free[pt] += 1
            self.occupancy[pt] = self.hit[pt] / float(self.hit[pt] + self.free[pt])
        pt = coords[-1]
        self.hit[pt] += 1
        self.occupancy[pt] = self.hit[pt] / float(self.hit[pt] + self.free[pt])

    def visualize_map(self, fname=None):
        x = self.x_limits[0]
        y = self.y_limits[0]

        x_count = int(math.ceil((self.x_limits[1] - self.x_limits[0]) / self.resolution))
        y_count = int(math.ceil((self.y_limits[1] - self.y_limits[0]) / self.resolution))

        occ = np.zeros((x_count+1, y_count+1))
        
        xid = 0
        while x < self.x_limits[1]:
            yid = y_count-1
            while yid >= 0:
                occ[(xid, yid)] = self.occupancy[self.to_grid((x, y))]
                yid -= 1
                y += self.resolution
            y = self.y_limits[0]
            xid += 1
            x += self.resolution

        plt.clf()
        plt.imshow(occ.transpose(), cmap="Greys")
        plt.colorbar()

        if fname is not None:
            plt.savefig(fname, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.show()


def create_occupancy_grid_map(dataset, resolution=0.1):
    """Generates an occupancy grid map from the data.

    The occupancy values are obtained using the simple counting method.

    :param fname path to the carmen log file to parse
    :param resolution the cell size to use
    """
    # Determine size of the gridmap
    scan_endpoints = []
    for data, label in data_generator(dataset["poses"], dataset["scans"]):
        scan_endpoints.extend(data)
    xlim, ylim = bounding_box(scan_endpoints, 10)

    # Build actual gridmap
    gridmap = OccupancyGridmap(xlim, ylim, 0.1)
    for pose, scan in zip(dataset["poses"], dataset["scans"]):
        gridmap.add(pose, scan)

    return gridmap
