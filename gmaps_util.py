import settings
import os
import gpxpy
import json
from PIL import Image, ImageDraw
import io
import requests
import math
from util import calculate_distance, lat_lng_to_world_coords
from static_map import StaticMap

# Function to snap points to the nearest road using Google Maps API
def snap_to_road(gmaps, points):
    snapped_points = []
    i = 0
    while i < len(points):  # API limit of 100 points per request
        prev_i = i
        chunk = []
        next_flight = False
        while i < len(points) and len(chunk) < 100:
            # Check whether this point has a distance > 50km from the previous point. In that case it's a flight segment
            # and we should not snap the points
            if i > 0 and calculate_distance(points[i - 1], points[i]) > 50:
                next_flight = True
                break
            chunk.append(points[i])
            i += 1
        response = gmaps.snap_to_roads(path=[(p.latitude, p.longitude) for p in chunk], interpolate=True)
        result = []
        interpolated_intermediate_points = []
        j = 0
        while j < len(response):
            point = response[j]
            if 'originalIndex' in point:
                result.extend(interpolated_intermediate_points)
                result.append((point["location"]["latitude"], point["location"]["longitude"], point["originalIndex"] + prev_i))
                interpolated_intermediate_points = []
            else:
                interpolated_intermediate_points.append((point["location"]["latitude"], point["location"]["longitude"], None))
            j += 1
        snapped_points.extend(result)

        if next_flight:
            snapped_points.append((points[i].latitude, points[i].longitude, i))
            i += 1

    return snapped_points

# Function to create the actual route as GPX track points from the snapped points
# We interpolate the time of snapped intermediate points based on the time of the previous and next correctly snapped points
# We already identified flight segments before so those should be in the snapped route already
# However there are some points google maps skips because it thinks they are just "zig zagging noise"
# We should however still include them in the final route. Also if a snapped point is further than 50m from the actual point
# we should discard it and all the intermediate points between the previous correctly snapped point and the current one
def postprocess_snapped_route(snapped_route, gpx):
    route = []
    route_since_last_correctly_snapped = []
    lastOriginalIndex = None
    for lat, lon, originalIndex in snapped_route:
        if len(route) == 0:
            route.append(gpx.tracks[0].segments[0].points[0])
            continue
        if lastOriginalIndex is not None and originalIndex is not None and originalIndex - lastOriginalIndex > 1:
            # We skipped some points, include them and discard intermediate points
            route_since_last_correctly_snapped = []
            while lastOriginalIndex < originalIndex:
                lastOriginalIndex += 1
                route.append(gpx.tracks[0].segments[0].points[lastOriginalIndex])
        elif originalIndex is not None:
            if calculate_distance(gpx.tracks[0].segments[0].points[originalIndex], (lat, lon)) < 0.2:
                lastOriginalIndex = originalIndex
                lastOriginal = route[-1]
                # add gpx track points with interpolated time for route_since_last_correctly_snapped and add them to route
                if len(route_since_last_correctly_snapped) > 0:
                    total_intermediate_points_distance = 0
                    for i in range(1, len(route_since_last_correctly_snapped)):
                        total_intermediate_points_distance += calculate_distance(route_since_last_correctly_snapped[i - 1], route_since_last_correctly_snapped[i])
                    total_intermediate_points_distance += calculate_distance(route_since_last_correctly_snapped[-1], (lat, lon))
                    total_time_diff = gpx.tracks[0].segments[0].points[lastOriginalIndex].time - lastOriginal.time
                    for i in range(1, len(route_since_last_correctly_snapped)):
                        # Calculate the time of the intermediate point based on the distance from the previous correctly snapped point
                        distance = calculate_distance(route_since_last_correctly_snapped[i - 1], route_since_last_correctly_snapped[i])
                        time_fraction = distance / total_intermediate_points_distance
                        time = lastOriginal.time + total_time_diff * time_fraction
                        gpx_point = gpxpy.gpx.GPXTrackPoint(latitude=route_since_last_correctly_snapped[i][0], longitude=route_since_last_correctly_snapped[i][1], time=time)
                        route.append(gpx_point)
                    # Add the last point
                    gpx_snapped_original = gpxpy.gpx.GPXTrackPoint(latitude=lat, longitude=lon, time=gpx.tracks[0].segments[0].points[originalIndex].time)
                    route.append(gpx_snapped_original)
                else:
                    route.append(gpx.tracks[0].segments[0].points[originalIndex])
                route_since_last_correctly_snapped = []
        else:
            route_since_last_correctly_snapped.append((lat, lon))
    return route

# Function to save snapped points to a file
def save_snapped_points(snapped_points, file_path):
    with open(file_path, 'w') as f:
        json.dump(snapped_points, f)

# Function to load snapped points from a file
def load_snapped_points(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Get a bounding box for the animation in center, width, height format
def get_animation_bbox(route):
    world_coords_route = [lat_lng_to_world_coords(point.latitude, point.longitude) for point in route]
    min_x = min(world_coords_route, key=lambda x: x[0])[0]
    max_x = max(world_coords_route, key=lambda x: x[0])[0]
    min_y = min(world_coords_route, key=lambda x: x[1])[1]
    max_y = max(world_coords_route, key=lambda x: x[1])[1]
    # Check if the first point that doesn't have x_min or x_max as x is between them, otherwise
    # swap them
    # For only two points we assume the shorter of the two routes was taken
    if len(world_coords_route) == 2:
        if max_x - min_x > 128:
            min_x, max_x = max_x, min_x
    else:
        for (x, _) in world_coords_route:
            if x != min_x and x != max_x:
                if x < min_x or x > max_x:
                    min_x, max_x = max_x, min_x
                break
    return ((min_x, max_x), (min_y, max_y))

# Desired image dimensions
output_width = 1920
output_height = 1080

def calculate_optimal_zoom(bbox, output_width, output_height):
    """Calculate the optimal zoom level for a bounding box and output dimensions."""

    if bbox[0][0] > bbox[0][1]:
        bbox = ((bbox[0][0], bbox[0][1] + 256), bbox[1])

    for z in range(21, -1, -1):  # Iterate from max zoom level (usually 21) to 0
        (left_x, top_y) = world_coords_to_tile_coords(bbox[0][0], bbox[1][0], z)
        (right_x, bottom_y) = world_coords_to_tile_coords(bbox[0][1], bbox[1][1], z)
        
        # Calculate width and height in pixels
        pixel_width = (abs(right_x - left_x) + 1) * 256
        pixel_height = (abs(bottom_y - top_y) + 1) * 256
        
        # Check if the pixel dimensions fit within the desired output dimensions
        if pixel_width <= output_width and pixel_height <= output_height:
            return z + 1  # This is the optimal zoom level
    
    return 0  # Default to zoom level 0 if none fit


# Calculate the optimal zoom level and tile coordinates
def world_coords_to_tile_coords(x, y, zoom, round_up=False):
    """Convert latitude and longitude to tile coordinates."""
    n = 2.0 ** zoom
    tile_x = n * (x / 256)
    tile_y = n * (y / 256)
    if round_up:
        tile_x = math.floor(tile_x)
        tile_y = math.floor(tile_y)
    return int(tile_x), int(tile_y)


def strech_bbox(bbox, aspect_ratio):
    """Stretch a bounding box to match a given aspect ratio."""
    ((x_min, x_max), (y_min, y_max)) = bbox

    # if x_min > x_max add 256 to x_max
    if x_min > x_max:
        x_max += 256

    # Calculate the original width and height
    original_width = x_max - x_min
    original_height = y_max - y_min
    original_aspect_ratio = original_width / original_height

    if original_aspect_ratio > aspect_ratio:
        # The original bounding box is wider than the target aspect ratio
        new_width = original_width
        new_height = new_width / aspect_ratio
    else:
        # The original bounding box is taller (or equal) than the target aspect ratio
        new_height = original_height
        new_width = new_height * aspect_ratio

    # Calculate the new bounding box coordinates, centered around the original one
    new_x_min = x_min - (new_width - original_width) / 2
    new_x_max = x_max + (new_width - original_width) / 2
    new_y_min = y_min - (new_height - original_height) / 2  # Adjust y_min up
    new_y_max = y_max + (new_height - original_height) / 2  # Adjust y_max down

    # Make sure the new bounding box is within the world bounds
    new_x_min = new_x_min % 256
    new_x_max = new_x_max % 256
    new_y_min = max(0, min(256, new_y_min))
    new_y_max = max(0, min(256, new_y_max))

    return ((new_x_min, new_x_max), (new_y_min, new_y_max))

    
def relative_position(outer_center, outer_size, inner_center, inner_size):
    # Calculate the top-left position of the outer rectangle in 1D
    outer_top_left = outer_center - outer_size / 2
    
    # Calculate the top-left position of the inner rectangle in 1D
    inner_top_left = inner_center - inner_size / 2
    
    # Calculate the relative position
    relative_position = inner_top_left - outer_top_left
    
    return relative_position

# Define function to fetch tiles
def fetch_tile(x, y, zoom):
    """Fetch a tile from Google Maps API."""
    # First lookup local tile cache
    # If not found, fetch from Google Maps API
    if os.path.exists(settings.output_directory + f"tile_cache/{zoom}/{x}_{y}.png"):
        return Image.open(settings.output_directory + f"tile_cache/{zoom}/{x}_{y}.png")
    else:
        url = f"https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={zoom}&key={settings.google_maps_api_key}"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch tile {x}, {y}, {zoom}")
        img = Image.open(io.BytesIO(response.content))
        os.makedirs(settings.output_directory + f"tile_cache/{zoom}/", exist_ok=True)
        img.save(settings.output_directory + f"tile_cache/{zoom}/{x}_{y}.png")
        return img

def fetch_map(bbox, size):
    # Determine zoom level based on desired output dimensions and bounding box
    # A more complex function would calculate the optimal zoom level.
    zoom = calculate_optimal_zoom(bbox, size[0], size[1])

    # Calculate the world coordinates of the bounding box (0-256)

    # Make sure bounding box has correct aspect ratio (while ensuring we don't go outside the world bounds)
    # Keep in mind that right can be less than left if the bounding box crosses the antimeridian
    aspect_ratio = size[0] / size[1]
    
    ((left_world, right_world), (top_world, bottom_world)) = strech_bbox(bbox, aspect_ratio)

    right_world_larger = right_world
    if right_world < left_world:
        right_world_larger += 256

    # Add 10% both height and width to the bounding box

    left_world -= (right_world_larger - left_world) * 0.05
    right_world_larger += (right_world_larger - left_world) * 0.05
    right_world = right_world_larger % 256

    # Find the maximum number you can add to the top and bottom without going outside the world bounds
    max_top = top_world / (bottom_world - top_world)
    max_bottom = (256 - bottom_world) / (bottom_world - top_world)
    max_addition = min(max_top, max_bottom)
    height_addition = min(0.05, max_addition)

    top_world -= (bottom_world - top_world) * height_addition
    bottom_world += (bottom_world - top_world) * height_addition

    # Calculate the tile range for the bounding box
    top_left_tile = world_coords_to_tile_coords(left_world, top_world, zoom)
    bottom_right_tile_theoretical = world_coords_to_tile_coords(right_world_larger, bottom_world, zoom)

    # Create a new blank image to paste tiles into
    num_tiles_x = bottom_right_tile_theoretical[0] - top_left_tile[0] + 1
    num_tiles_y = bottom_right_tile_theoretical[1] - top_left_tile[1] + 1
    output_image = Image.new("RGB", (num_tiles_x * 256, num_tiles_y * 256))

    # Fetch and stitch tiles
    for x in range(top_left_tile[0], bottom_right_tile_theoretical[0] + 1):
        x_actual = x % (2 ** zoom)
        for y in range(top_left_tile[1], bottom_right_tile_theoretical[1] + 1):
            tile = fetch_tile(x_actual, y, zoom)
            output_image.paste(tile, ((x - top_left_tile[0]) * 256, (y - top_left_tile[1]) * 256))

    # Now the image is too large, so we need to crop it to the desired output dimensions (1920x1080) such that the center of the bbox is in the center of the image
    # Calculate the center of the bbox in world coords
    bbox_center_x = (left_world + right_world) / 2
    bbox_center_y = (top_world + bottom_world) / 2
    bbox_width = right_world_larger - left_world
    bbox_height = bottom_world - top_world
    
    # Get world_coords of of the image
    top_left_x = top_left_tile[0] / (2 ** zoom) * 256
    top_left_y = top_left_tile[1] / (2 ** zoom) * 256
    bottom_right_x = (bottom_right_tile_theoretical[0] + 1) / (2 ** zoom) * 256
    bottom_right_y = (bottom_right_tile_theoretical[1] + 1) / (2 ** zoom) * 256
    image_width = bottom_right_x - top_left_x
    image_height = bottom_right_y - top_left_y

    # We need to cut the image such that the center of the bbox is in the center of the image
    # and such that image width and height are the same as the bbox width and height
    # Calculate where the bbox top left corner is in the image
    bbox_left_px = abs(top_left_x - left_world) / image_width * output_image.width
    bbox_top_px = abs(top_left_y - top_world) / image_height * output_image.height
    bbox_right_px = bbox_left_px + bbox_width / image_width * output_image.width
    bbox_bottom_px = bbox_top_px + bbox_height / image_height * output_image.height

    output_image = output_image.resize(size, box=(int(bbox_left_px), int(bbox_top_px), int(bbox_right_px), int(bbox_bottom_px)), resample=Image.Resampling.LANCZOS)


    return StaticMap((bbox_center_x, bbox_center_y), (right_world_larger - left_world, bottom_world - top_world), output_image)


def get_minimap(position, zoom_lvl=10):
    # Calculate the tile coordinates of the center of the minimap
    world_coords_pos = lat_lng_to_world_coords(position[0], position[1])
    center_tile = world_coords_to_tile_coords(world_coords_pos[0], world_coords_pos[1], zoom_lvl)

    extra_tiles = math.ceil(settings.minimap_width / 256 / 2)
    total_tile_width = extra_tiles * 2 + 1


    # Fetch the tiles for the minimap
    minimap = Image.new("RGB", (256 * total_tile_width, 256 * total_tile_width))
    for x in range(center_tile[0] - extra_tiles, center_tile[0] + extra_tiles + 1):
        for y in range(center_tile[1] - extra_tiles, center_tile[1] + extra_tiles + 1):
            tile = fetch_tile(x % (2 ** zoom_lvl), y % (2 ** zoom_lvl), zoom_lvl)
            minimap.paste(tile, ((x - center_tile[0] + extra_tiles) * 256, (y - center_tile[1] + extra_tiles) * 256))

    center_tile_world_x = center_tile[0] / (2 ** zoom_lvl) * 256
    center_tile_world_y = center_tile[1] / (2 ** zoom_lvl) * 256
    tile_width = 256 / (2 ** zoom_lvl)

    center = (center_tile_world_x + tile_width / 2, center_tile_world_y + tile_width / 2)

    # Crop the minimap to 256x256
    static_map = StaticMap(center, (256 * (total_tile_width / (2 ** zoom_lvl)), 256 * (total_tile_width / (2 ** zoom_lvl))), minimap)

    # Get the pixel coordinates of the position
    pixel_coords = static_map.to_px(world_coords_pos[0], world_coords_pos[1])

    minimap_size = settings.minimap_width

    # Cut a 256x256 square around the position
    #minimap = minimap.crop((int(pixel_coords[0] - 128), int(pixel_coords[1] - 128), int(pixel_coords[0] + 128), int(pixel_coords[1] + 128)))

    # cut a minimap_sizexminimap_size square around the position
    minimap = minimap.crop((int(pixel_coords[0] - minimap_size / 2), int(pixel_coords[1] - minimap_size / 2), int(pixel_coords[0] + minimap_size / 2), int(pixel_coords[1] + minimap_size / 2)))

    # Draw a red circle around the position
    minimap_draw = ImageDraw.Draw(minimap)
    dot_size = 3
    minimap_draw.ellipse((minimap_size / 2 - dot_size, minimap_size / 2 - dot_size, minimap_size / 2 + dot_size, minimap_size / 2 + dot_size), fill="red")

    return minimap


def test_get_minimap():
    position = (32.670978,-117.241656)
    minimap = get_minimap(position)
    minimap.save(settings.output_directory + "minimap.png")

if __name__ == "__main__":
    test_get_minimap()