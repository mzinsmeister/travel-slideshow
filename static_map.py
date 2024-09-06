import json
import gpxpy
from moviepy.editor import VideoClip
import numpy as np
from PIL import Image, ImageDraw
import PIL
from util import calculate_distance, lat_lng_to_world_coords
import numpy as np

class StaticMap:
    def __init__(self, center: tuple[float, float], world_coord_size: tuple[float, float], image: Image):
        self.center = center
        self.image = image
        self.world_coord_size = world_coord_size

    # Todo: Handle anti-meridian
    def to_px(self, x, y):
        """Convert world coordinates to pixel coordinates. (0, 0) is the top left corner."""
        x0, y0 = self.center
        w, h = self.world_coord_size
        px = (x - x0 + w / 2) / w * self.image.width
        py = (y0 - y + h / 2) / h * self.image.height
        py = self.image.height - py
        return (px, py)
    
    
    def __repr__(self) -> str:
        return f"StaticMap(center={self.center}, world_coord_size={self.world_coord_size}, size={(self.image.width, self.image.height)})"

    def add_route(self, draw, route):
        world_route = [lat_lng_to_world_coords(point.latitude, point.longitude) for point in route]
        path = []
        for point in world_route:
            px, py = self.to_px(*point)
            path.append((px, py))
        draw.line(path, fill="blue", width=5)
        return draw

    # draw route on the map
    def draw_route(self, route):
        img = self.image.copy()
        draw = ImageDraw.Draw(img)
        self.add_route(draw, route)
        return img
    
    def create_route_animation(self, route: list, route_segment_bounds: tuple[int, int], flight_segments, output_path, km_per_second=30):

        world_route = [lat_lng_to_world_coords(point.latitude, point.longitude) for point in route]

        route_segment = route[route_segment_bounds[0]:route_segment_bounds[1] + 1]
        route_segment_world = world_route[route_segment_bounds[0]:route_segment_bounds[1] + 1]

        route_distance = sum([calculate_distance(route_segment[i], route_segment[i + 1]) for i in range(len(route_segment) - 1)])
        duration = route_distance / km_per_second

        # to travel at constant speed, we need to calculate how long each segment should take
        segment_duration = [calculate_distance(route_segment[i], route_segment[i + 1]) / km_per_second for i in range(len(route_segment) - 1)]


        def make_frame(t):
            img = self.image.copy()
            draw = ImageDraw.Draw(img)

            # find the segment that we are currently on
            segment = 0
            total_time = 0
            while segment < len(segment_duration) and total_time + segment_duration[segment] < t:
                total_time += segment_duration[segment]
                segment += 1

            # Draw the route before the animation starts
            for (i, f) in enumerate(flight_segments):
                if f[0] > route_segment_bounds[0]:
                    # Draw the rest of the driving route
                    if i > 0:
                        start = flight_segments[i - 1][1]
                        self.add_route(draw, route[start:route_segment_bounds[0] + 1])
                    break

                # Draw the route up to the current flight segment
                if i == 0:
                    start = 0
                else:
                    start = flight_segments[i - 1][1]
                
                self.add_route(draw, route[start:f[0]])

                # Draw the flight path if either the start or end of the segment is within the current frame
                f_start = route[f[0]]
                f_world_start = np.array(world_route[f[0]])
                f_end = route[f[1]]
                f_world_end = np.array(world_route[f[1]])
                start_in_frame =  np.all(np.abs(f_world_start - self.center) <= np.array(self.world_coord_size) / 2)
                end_in_frame = np.all(np.abs(f_world_end - self.center) <= np.array(self.world_coord_size) / 2)
                if start_in_frame or end_in_frame:
                    self.add_flight_path(draw, f_start, f_end)


            # insert the current route sector up to the current point
            self.add_route(draw, route[route_segment_bounds[0]:route_segment_bounds[0] + segment + 1])            

            if segment_duration[segment] != 0:
                # calculate the progress within the segment
                segment_progress = (t - total_time) / segment_duration[segment]
                # calculate the position of the current point
                end = np.add(route_segment_world[segment], segment_progress * np.subtract(route_segment_world[segment + 1], route_segment_world[segment]))
                px, py = self.to_px(*end)

                last = self.to_px(*route_segment_world[segment])

                draw.line([last, (px, py)], fill="blue", width=5)
            else:
                px, py = self.to_px(*route_segment_world[segment])
            
            # draw the current point
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill="red")
            return np.array(img)

        
        animation = VideoClip(make_frame, duration=duration)
        animation.write_videofile(output_path, fps=24)

    def add_flight_path(self, draw, start, end):
        world_x, world_y = lat_lng_to_world_coords(start.latitude, start.longitude)
        x1, y1 = self.to_px(world_x, world_y)
        world_x, world_y = lat_lng_to_world_coords(end.latitude, end.longitude)
        x2, y2 = self.to_px(world_x, world_y)    

        #draw.line((x1, y1, x2, y2), fill="blue", width=5)
        # Draw an arc instead of a line

        # Control point for the Bezier curve (for the arc)
        world_x, world_y = lat_lng_to_world_coords(start.latitude, start.longitude)
        x1, y1 = self.to_px(world_x, world_y)
        world_x, world_y = lat_lng_to_world_coords(end.latitude, end.longitude)
        x2, y2 = self.to_px(world_x, world_y)    

        #draw.line((x1, y1, x2, y2), fill="blue", width=5)
        # Draw an arc instead of a line

        # Control point for the Bezier curve (for the arc)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 - 150  # Adjust the -150 for more or less arc

        # Calculate the number of dashes to draw by dividing the arc length by the dash length


        # Generate points along the Bezier curve
        t_values = np.linspace(0, 1, num=1000)
        arc_points = [
            (
                (1 - t)**2 * x1 + 2 * (1 - t) * t * cx + t**2 * x2,
                (1 - t)**2 * y1 + 2 * (1 - t) * t * cy + t**2 * y2
            )
            for t in t_values
        ]

        # Convert points to integer tuples
        arc_points = [(int(x), int(y)) for x, y in arc_points]

        # Split up the arc points into dashes (one per n pixels)
        dash_length = 20
        dashes = []
        current_start = 0
        visible = True
        for i in range(1, len(arc_points)):
            current_len = np.linalg.norm(np.subtract(arc_points[i], arc_points[current_start]))
            if current_len > dash_length or (i == len(arc_points) - 1 and current_len > 0):
                if visible:
                    dashes.append(arc_points[current_start:i + 1])
                current_start = i
                visible = not visible

        dashes = [[dash[0], dash[-1]] for dash in dashes]

        for dash in dashes:
            draw.line(dash, fill="green", width=5)
    
    # draw flight path as an arch
    def draw_flight_path(self, start, end):
        img = self.image.copy()
        draw = ImageDraw.Draw(img)
        self.add_flight_path(draw, start, end)
        return img
        

    # Function to render an animation of a flight segment
    def create_flight_animation(self, start, end, output_path):
        duration = 3 # Adjust as needed

        x1, y1 = self.to_px(*lat_lng_to_world_coords(start.latitude, start.longitude))
        x2, y2 = self.to_px(*lat_lng_to_world_coords(end.latitude, end.longitude))

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 - 150  # Adjust the -150 for more or less arc

        # Generate points along the Bezier curve
        t_values = np.linspace(0, 1, num=1000)
        arc_points = [
            (
                (1 - t)**2 * x1 + 2 * (1 - t) * t * cx + t**2 * x2,
                (1 - t)**2 * y1 + 2 * (1 - t) * t * cy + t**2 * y2
            )
            for t in t_values
        ]

        # Convert points to integer tuples
        arc_points = [(int(x), int(y)) for x, y in arc_points]

        # Split up the arc points into dashes (one per n pixels)
        dash_length = 20
        dashes = []
        current_start = 0
        for i in range(1, len(arc_points)):
            current_len = np.linalg.norm(np.subtract(arc_points[i], arc_points[current_start]))
            if current_len > dash_length or (i == len(arc_points) - 1 and current_len > 0):
                dashes.append(arc_points[current_start:i + 1])
                current_start = i

        # Split up the arc points into dashes (one per n pixels)
        dash_length = 20
        dashes = []
        current_start = 0
        total_length = 0
        for i in range(1, len(arc_points)):
            current_len = np.linalg.norm(np.subtract(arc_points[i], arc_points[current_start]))
            if current_len > dash_length or (i == len(arc_points) - 1 and current_len > 0):
                dashes.append(arc_points[current_start:i + 1])
                current_start = i
                total_length += current_len

        dashes = [[dash[0], dash[-1]] for dash in dashes]

        # Calculate the start time of the last dash according to its length
        per_px_duration = (duration - 0.1) / total_length
        last_dash_duration = per_px_duration * np.linalg.norm(np.subtract(dashes[-1][0], dashes[-1][1]))
        last_dash_start_time = (duration - 0.1) - last_dash_duration

        def make_frame(t):
            img = self.image.copy()
            draw = ImageDraw.Draw(img)

            if t >= last_dash_start_time:
                current_dash_index = len(dashes) - 1
                # We let the animation run a tiny bit faster than necessary to make sure the last dash is fully drawn
                current_dash_progress = min((t - last_dash_start_time) / last_dash_duration, 0.9999999)
            else:
                progress = t / last_dash_start_time
                current_dash_index = int(progress * (len(dashes) - 1))
                current_dash_start_progress = current_dash_index / (len(dashes) - 1)
                next_dash_start_progress = (current_dash_index + 1) / (len(dashes) - 1)
                current_dash_progress_range = next_dash_start_progress - current_dash_start_progress
                current_dash_progress = (progress - current_dash_start_progress) / current_dash_progress_range

            for i, dash in enumerate(dashes[:current_dash_index]):
                if i % 2 == 0:
                    draw.line(dash, fill="green", width=5)

            current_dash = dashes[current_dash_index]
            current_dash_segment = int(current_dash_progress * (len(current_dash) - 1))
            current_point = current_dash[current_dash_segment]

            current_point_start_progress = current_dash_segment / (len(current_dash) - 1)
            next_point_start_progress = (current_dash_segment + 1) / (len(current_dash) - 1)
            inner_progress = (current_dash_progress - current_point_start_progress) / (next_point_start_progress - current_point_start_progress)
            next_point = current_dash[current_dash_segment + 1]
            interpolated = np.add(current_point, inner_progress * np.subtract(next_point, current_point))

            if current_dash_index % 2 == 0:
                draw.line(current_dash[:current_dash_segment], fill="green", width=5)
                # Draw interpolated section of the next segment within the current dash
                draw.line([current_point, tuple(interpolated)], fill="green", width=5)
                
                
            draw.ellipse((interpolated[0] - 5, interpolated[1] - 5, interpolated[0] + 5, interpolated[1] + 5), fill="red")


            return np.array(img)

        animation = VideoClip(make_frame, duration=duration)
        animation.write_videofile(output_path, fps=24)


def test_draw_route():
    # Test with route segment from 1339 to 1340 of the final.gpx route
    output_directory = 'data/tmp/'
    with open(output_directory + 'final.gpx', 'r') as file:
        gpx = gpxpy.parse(file)
    route = [point for track in gpx.tracks for segment in track.segments for point in segment.points]
    segment = (1834, 2070)
    world_coord_size = (2.0549912450275016, 1.1559325753279666)
    center=(44.90427872204651, 102.83516537291779)
    image = Image.open('data/tmp/driving_segment.png') 
    static_map = StaticMap(center, world_coord_size, image)
    with_route = static_map.draw_route(route[segment[0]:segment[1]])
    with_route.save(output_directory + 'route.png')

def test_draw_flight():
        # Test with route segment from 1339 to 1340 of the final.gpx route
    output_directory = 'data/tmp/'
    with open(output_directory + 'final.gpx', 'r') as file:
        gpx = gpxpy.parse(file)
    route = [point for track in gpx.tracks for segment in track.segments for point in segment.points]
    segment = (1339, 1340)
    center=(101.03351013422223, 91.8834101349204)
    world_coord_size=(78.143554384, 43.955749341)
    image = Image.open('data/tmp/flight_segment.png') 
    static_map = StaticMap(center, world_coord_size, image)
    world_1 = lat_lng_to_world_coords(route[segment[0]].latitude, route[segment[0]].longitude)
    world_2 = lat_lng_to_world_coords(route[segment[1]].latitude, route[segment[1]].longitude)
    print(world_1, world_2)
    with_route = static_map.draw_flight_path(route[segment[0]], route[segment[1]])
    with_route.save(output_directory + 'flight.png')

def test_create_route_animation():
    output_directory = 'data/tmp/'
    with open(output_directory + 'final.gpx', 'r') as file:
        gpx = gpxpy.parse(file)
    route = [point for track in gpx.tracks for segment in track.segments for point in segment.points]
    with open(output_directory + 'flight_segments.json', 'r') as file:
        flight_segments = json.load(file)
    segment = (1834, 2070)
    world_coord_size = (2.0549912450275016, 1.1559325753279666)
    center=(44.90427872204651, 102.83516537291779)
    image = Image.open('data/tmp/driving_segment.png') 
    static_map = StaticMap(center, world_coord_size, image)
    static_map.create_route_animation(route, segment, flight_segments, output_directory + 'route.mp4')

def test_create_flight_animation():
    output_directory = 'data/tmp/'
    with open(output_directory + 'final.gpx', 'r') as file:
        gpx = gpxpy.parse(file)
    route = [point for track in gpx.tracks for segment in track.segments for point in segment.points]
    segment = (1339, 1340)
    center=(101.03351013422223, 91.8834101349204)
    world_coord_size=(78.143554384, 43.955749341)
    image = Image.open('data/tmp/flight_segment.png') 
    static_map = StaticMap(center, world_coord_size, image)
    static_map.create_flight_animation(route[segment[0]], route[segment[1]], output_directory + 'flight.mp4')

if __name__ == "__main__":
    test_create_flight_animation()
