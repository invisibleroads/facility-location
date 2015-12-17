import numpy as np
from random import uniform
from scipy.optimize import minimize


def place_hotspot_points(
        interest_point_tree, compute_distance,
        hotspot_point_count, hotspot_point_radius,
        hotspot_point_xy_min, hotspot_point_xy_max, iteration_count):
    interest_point_count = interest_point_tree.n

    def compute_loss(hotspot_points):
        # Restore hotspot_points structure because minimize flattens it
        hotspot_points = np.array(hotspot_points).reshape((
            hotspot_point_count, 2))
        loss = 0
        for hotspot_point in hotspot_points:
            distances = filter(lambda x: x < np.inf, interest_point_tree.query(
                hotspot_point, interest_point_count)[0])
            # Be close to many interest points
            # Minimize the sum of distances to interest point
            loss += sum(distances)
            # Balance travel time to each interest point
            # Minimize the maximum distance to any interest point
            loss += max(distances)
            # Have good coverage
            # Maximize distance from other hotspot_points if it in range
            if min(distances) < hotspot_point_radius:
                loss -= sum(compute_distance(
                    hotspot_point, x) for x in hotspot_points)
        return loss

    hotspot_point_x_bounds, hotspot_point_y_bounds = _get_xy_bounds(
        hotspot_point_xy_min, hotspot_point_xy_max)
    make_values = lambda: [(
        uniform(*hotspot_point_x_bounds), uniform(*hotspot_point_y_bounds),
    ) for x in xrange(hotspot_point_count)]
    return get_good_values(
        compute_loss, make_values, iteration_count,
    ).reshape((hotspot_point_count, 2))


def get_good_values(compute_loss, make_values, iteration_count):
    good_loss, good_values = np.inf, None
    for iteration_index in xrange(iteration_count):
        optimization = minimize(compute_loss, make_values())
        if optimization['fun'] < good_loss:
            good_loss = optimization['fun']
            good_values = optimization['x']
    return good_values


def _get_xy_bounds(xy_min, xy_max):
    x_min, y_min = xy_min
    x_max, y_max = xy_max
    x_bounds = x_min, x_max
    y_bounds = y_min, y_max
    return x_bounds, y_bounds
