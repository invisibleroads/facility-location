from argparse import ArgumentParser
from crosscompute_table import TableType
from invisibleroads_macros.disk import make_enumerated_folder_for, make_folder
from invisibleroads_macros.log import format_summary
from os.path import join
from pandas import DataFrame
from pysal.cg.kdtree import Arc_KDTree
from pysal.cg.sphere import arcdist, RADIUS_EARTH_KM
from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean

from facility_location import place_hotspot_points


EARTH_RADIUS_IN_METERS = RADIUS_EARTH_KM * 1000
HOTSPOT_COUNT_INVALID = 'hotspot_point_count.error = must be at least one'
HOTSPOT_RADIUS_INVALID = 'hotspot_point_radius_in_meters.error = must be greater than zero'  # noqa


def run(
        target_folder, interest_point_table,
        interest_point_table_x_column, interest_point_table_y_column,
        hotspot_point_count, hotspot_point_radius_in_meters):
    # Make sure that we have (longitude, latitude) for (x, y)
    if _is_latitude(interest_point_table_x_column) and _is_longitude(
            interest_point_table_y_column):
        interest_point_table_x_column, interest_point_table_y_column = \
            interest_point_table_y_column, interest_point_table_x_column
    # Get interest_points
    interest_points = interest_point_table[[
        interest_point_table_x_column, interest_point_table_y_column]].values
    # Use appropriate structures
    if _is_longitude(interest_point_table_x_column) and _is_latitude(
            interest_point_table_y_column):
        interest_point_tree = Arc_KDTree(
            interest_points, radius=EARTH_RADIUS_IN_METERS)
        compute_distance = lambda a, b: arcdist(
            a, b, radius=EARTH_RADIUS_IN_METERS)
    else:
        interest_point_tree = KDTree(interest_points)
        compute_distance = euclidean
    # Place
    hotspot_points = place_hotspot_points(
        interest_point_tree, compute_distance,
        hotspot_point_count, hotspot_point_radius_in_meters,
        interest_points.min(axis=0), interest_points.max(axis=0),
        iteration_count=3)
    # Save
    hotspot_point_table = _get_hotspot_point_table(
        hotspot_points, interest_point_table,
        interest_point_table_x_column, interest_point_table_y_column)
    hotspot_point_table_path = join(target_folder, 'hotspot_points.csv')
    hotspot_point_table.to_csv(hotspot_point_table_path, index=False)
    return [
        ('hotspot_point_table_path', hotspot_point_table_path),
    ]


def _is_longitude(x):
    return x.lower().startswith('lon')


def _is_latitude(x):
    return x.lower().startswith('lat')


def _get_hotspot_point_table(
        hotspot_points, interest_point_table,
        interest_point_table_x_column, interest_point_table_y_column):
    'Make table from points and order columns to match original table'
    hotspot_point_table = DataFrame(hotspot_points, columns=[
        interest_point_table_x_column, interest_point_table_y_column])

    interest_point_table_columns = list(interest_point_table.columns)
    x_column_index = interest_point_table_columns.index(
        interest_point_table_x_column)
    y_column_index = interest_point_table_columns.index(
        interest_point_table_y_column)

    if y_column_index < x_column_index:
        hotspot_point_table = hotspot_point_table[[
            interest_point_table_y_column, interest_point_table_x_column]]
    return hotspot_point_table


if __name__ == '__main__':
    argument_parser = ArgumentParser()
    argument_parser.add_argument(
        '--target_folder',
        metavar='FOLDER', type=make_folder)

    argument_parser.add_argument(
        '--interest_point_table_path',
        metavar='PATH', required=True)
    argument_parser.add_argument(
        '--interest_point_table_x_column',
        metavar='COLUMN', required=True)
    argument_parser.add_argument(
        '--interest_point_table_y_column',
        metavar='COLUMN', required=True)

    argument_parser.add_argument(
        '--hotspot_point_count',
        metavar='COUNT', type=int, required=True)
    argument_parser.add_argument(
        '--hotspot_point_radius_in_meters',
        metavar='RADIUS', type=float, required=True)

    args = argument_parser.parse_args()
    if args.hotspot_point_count < 1:
        exit(HOTSPOT_COUNT_INVALID)
    if args.hotspot_point_radius_in_meters <= 0:
        exit(HOTSPOT_RADIUS_INVALID)
    d = run(
        args.target_folder or make_enumerated_folder_for(__file__),

        TableType().load(
            args.interest_point_table_path),
        args.interest_point_table_x_column,
        args.interest_point_table_y_column,

        args.hotspot_point_count,
        args.hotspot_point_radius_in_meters)
    print(format_summary(d))
