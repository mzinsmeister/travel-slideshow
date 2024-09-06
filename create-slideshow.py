import datetime
import struct
from matplotlib import font_manager
import PIL.ExifTags
import PIL.ImageFont
import odf
import odf.namespaces
import odf.opendocument
import odf.style
import timezonefinder
import settings
import googlemaps
import googlemaps.client
import gpxpy
import json
from geopy.distance import geodesic
from geopy import Point, Location
import matplotlib.pyplot as plt
from moviepy.editor import VideoClip
import os
import numpy as np
from PIL import Image, ImageDraw
import PIL
import io
from dotenv import load_dotenv
import requests
import math
import pyproj
from pytz import timezone

from gmaps_util import fetch_map, get_minimap, load_snapped_points, postprocess_snapped_route, save_snapped_points, snap_to_road
from static_map import StaticMap
from util import calculate_distance, lat_lng_to_world_coords


# Function to parse the GPX file
def parse_gpx(file_path):
    with open(file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)
    return gpx

# If n consecutive track points have the same timestamp, we should instead interpolate the time using the first of them and the next one with a different timestamp
# Using the percentage of the distance between the two points to calculate the time of the intermediate points
def fix_gpx_times(gpx):
    for track in gpx.tracks:
        for segment in track.segments:
            i = 0
            while i < len(segment.points) - 1:
                j = i + 1
                while j < len(segment.points) and segment.points[j].time == segment.points[i].time:
                    j += 1
                if j < len(segment.points):
                    total_distance = calculate_distance(segment.points[i], segment.points[j])
                    total_time = segment.points[j].time - segment.points[i].time
                    for k in range(i + 1, j):
                        distance = calculate_distance(segment.points[i], segment.points[k])
                        time_fraction = distance / total_distance
                        segment.points[k].time = segment.points[i].time + total_time * time_fraction
                i = j
    return gpx


def to_geopy(point):
    return Point(point['latitude'], point['longitude'])


# Function to identify segments as road or flight
def identify_segments(waypoints, max_distance_km=300):
    segments = []
    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        end = waypoints[i + 1]
        distance = calculate_distance(start, end)
        if distance > max_distance_km:
            segments.append((start, end, 'flight'))
        else:
            segments.append((start, end, 'road'))
    return segments

# Helper function to convert a matplotlib figure to an image
def plt_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    return np.array(img)


# Function to add a video slide to the ODP presentation
def add_video_slide(doc: odf.opendocument.OpenDocumentPresentation, video_path, dpstyle, masterpage, titlestyle, videostyle):
    # <draw:frame draw:name="Media object 1" draw:style-name="gr1" draw:layer="layout"
    #     svg:width="50.799cm" svg:height="28.574cm" svg:x="-11.297cm" svg:y="-6.374cm">
    #     <draw:plugin xlink:href="Media/driving_1834_2070.mp4" xlink:type="simple"
    #         xlink:show="embed" xlink:actuate="onLoad" draw:mime-type="video/mp4">
    #         <draw:param draw:name="Loop" draw:value="false" />
    #         <draw:param draw:name="Mute" draw:value="false" />
    #         <draw:param draw:name="VolumeDB" draw:value="0" />
    #     </draw:plugin>
    # </draw:frame>

    page = odf.draw.Page(stylename=dpstyle, masterpagename=masterpage)
    doc.presentation.addElement(page)
    #titleframe = odf.draw.Frame(stylename=titlestyle, width="720pt", height="56pt", x="40pt", y="10pt")
    #page.addElement(titleframe)
    #textbox = odf.TextBox()
    #titleframe.addElement(textbox)
    #textbox.addElement(odf.P(text="test"))

    p = doc.addPicture(video_path, mediatype="video/mp4")

    videoframe = odf.draw.Frame(stylename=videostyle, width="1440pt", height="810pt", x="0pt", y="0pt")
    page.addElement(videoframe)
    video = odf.draw.Plugin(href=p)
    video.setAttrNS(odf.namespaces.XLINKNS, 'href', p)
    video.setAttrNS(odf.namespaces.XLINKNS, 'type', 'simple')
    video.setAttrNS(odf.namespaces.XLINKNS, 'show', 'embed')
    video.setAttrNS(odf.namespaces.XLINKNS, 'actuate', 'onLoad')
    video.setAttrNS(odf.namespaces.DRAWNS, 'mime-type', 'video/mp4')
    videoframe.addElement(video)#

def getImageInfo(data):

    img = PIL.Image.open(io.BytesIO(data))
    content_type = img.get_format_mimetype()
    img = PIL.ImageOps.exif_transpose(img)
    width, height = img.size

    return img, content_type, width, height

def add_black_bars(image, offset_x=None, offset_y=None):
    
    # Define the desired output size
    target_size = (1920, 1080)

    # Calculate the appropriate size to maintain aspect ratio
    image.thumbnail(target_size, Image.LANCZOS)

    # Create a new image with the target size and black background
    new_image = Image.new('RGB', target_size, (0, 0, 0))

    # Calculate the position to paste the resized image onto the black background
    paste_position = (
        (target_size[0] - image.width) // 2,
        (target_size[1] - image.height) // 2
    )

    if offset_x is not None:
        paste_position = (offset_x, paste_position[1])
    if offset_y is not None:
        paste_position = (paste_position[0], offset_y)

    # Paste the resized image onto the new image (with black bars)
    new_image.paste(image, paste_position)

    return new_image

def add_photo_slide(doc: odf.opendocument.OpenDocumentPresentation, photo_path, dpstyle, masterpage, titlestyle, photostyle, minimap, timestamp):
    with open(photo_path, 'rb') as f:
        pictdata = f.read()
    img, ct,orig_w,orig_h = getImageInfo(pictdata) # Get dimensions in pixels
    if ct != 'image/jpeg':
        return
    
    pres_w_pt = 1440
    pres_h_pt = 810

    minimap_w = settings.minimap_width
    minimap_h = 50 if minimap is None else minimap.height + 50

    # Calculate the aspect ratio of the image
    aspect_ratio = orig_w / orig_h
    # Calculate the aspect ratio of the presentation
    pres_aspect_ratio = pres_w_pt / pres_h_pt
    # Calculate the width and height of the image in the presentation
    if aspect_ratio > pres_aspect_ratio:
        # Image is wider than the presentation
        w_px = 1920
        h_px = int(orig_h * (w_px / orig_w))
        offset_x = 0
        offset_y = int(max((1080 - h_px) / 2 - minimap_h / 2, 0))
    else:
        # Image is taller than the presentation
        h_px = 1080
        w_px = int(orig_w * (h_px / orig_h))
        offset_y = 0
        offset_x = int(max((1920 - w_px) / 2 - minimap_w / 2, 0))

    # Pad the image with black bars if necessary        


    page = odf.draw.Page(stylename=dpstyle, masterpagename=masterpage)
    doc.presentation.addElement(page)
    # titleframe = odf.draw.Frame(stylename=titlestyle, width="720pt", height="56pt", x="40pt", y="10pt")
    # page.addElement(titleframe)
    # textbox = odf.draw.TextBox()
    # titleframe.addElement(textbox)
    # textbox.addElement(P(text=picture))
    photoframe = odf.draw.Frame(stylename=photostyle, width="%fpt" % pres_w_pt, height="%fpt" % pres_h_pt, x="%fpt" % 0, y="%fpt" % 0)
    page.addElement(photoframe)
    # Scale to the correct size
    img = img.resize((int(w_px), int(h_px)), Image.Resampling.LANCZOS)
    img = add_black_bars(img, offset_x, offset_y)
    # Add the minimap at 50px from the bottom right corner
    minimap_height = 0
    if minimap is not None:
        img.paste(minimap, (1920 - minimap.width, 1080 - minimap.height))
        minimap_height = minimap.height
    # Add the timestamp right above the minimap with a height of 50px and a width of minimap.width
    if timestamp is not None:
        draw = ImageDraw.Draw(img)
        # We could also draw a black rectangle but i think that's probably not necessary
        # draw.rectangle([1920 - settings.minimap_width, 1080 - minimap_height - 50, 1920, 1080 - minimap_height], fill=(0, 0, 0))
        font = font_manager.FontProperties(family='sans-serif', weight='bold')
        file = font_manager.findfont(font)
        font = PIL.ImageFont.truetype(file, 28)
        timestamp_text = timestamp.strftime("%d.%m.%Y %H:%M")
        draw.text((1920 - 10, 1080 - minimap_height - 10), timestamp_text, fill=(255, 255, 255), anchor='rb', align='center', font=font, stroke_fill=(0, 0, 0), stroke_width=2)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='jpeg')
    img_bytes.seek(0)
    img_bytes = img_bytes.read()
    href = doc.addPictureFromString(img_bytes, mediatype=ct)
    photoframe.addElement(odf.draw.Image(href=href))


def select_animation_segments(route, waypoints):
    flying_segments = []
    driving_segments = []
    # Determine flying segments by looking for consecutive track points with a distance > 300km
    for i in range(len(route) - 1):
        if calculate_distance(route[i], route[i + 1]) > 300:
            flying_segments.append((i, i + 1))
    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        end = waypoints[i + 1]
        distance = calculate_distance(start, end)
        if distance < 300 and distance > 25: # Ignore short distances and flights
            # find closest points
            start_i = 0
            start_point = route[0]
            start_distance = calculate_distance(start, start_point)
            end_i = 0
            end_point = route[0]
            first_end_i = None
            first_end_point = None
            end_distance = calculate_distance(end, end_point)
            #(start_i, start_point) = min([point for point in enumerate(route) if point[1].time < end.time], key=lambda point: calculate_distance(start, point[1]))
            # find end point which must be after start
            #(end_i, end_point) = min([point for point in enumerate(route) if point[1].time > start_point.time], key=lambda point: calculate_distance(end, point[1]))
            for j in range(1, len(route)):
                start_set = False
                if route[j].time < end.time:
                    start_distance_j = calculate_distance(start, route[j])
                    # We prefer later points even if they are slightly further away
                    # And generally also allow any point that is within 1% of the distance
                    if start_distance_j * 0.7 < start_distance or (start_distance_j < distance * 0.02):
                        start_distance = start_distance_j
                        start_i = j
                        start_point = route[j]  
                        end_point = route[j]
                        end_i = j
                        end_distance = calculate_distance(end, route[j])
                        first_end_i = None
                        first_end_point = None
                        start_set = True
                if not start_set:
                    end_distance_j = calculate_distance(end, route[j])
                    # Take first point that is within 2% of the distance
                    if j != start_i and first_end_i is None and end_distance_j < distance * 0.02:
                        first_end_i = j
                        first_end_point = route[j]
                    # We only replace the end point if it is closer to the previous
                    # closest point by at least 50% to prefer earlier points
                    factor = 0.5
                    if start_i == end_i:
                        factor = 1
                    if end_distance_j < end_distance * factor:
                        end_distance = end_distance_j
                        end_i = j
                        end_point = route[j]

            if first_end_i is not None:
                end_i = first_end_i
                end_point = first_end_point
            #print(f"Driving segment from {start.name} to {end.name}")
            driving_segments.append((start_i, end_i))

    
    return flying_segments, driving_segments

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


# Define function to fetch tiles
def fetch_tile(x, y, zoom):
    """Fetch a tile from Google Maps API."""
    url = f"https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={zoom}&key={google_maps_api_key}"
    response = requests.get(url)
    return Image.open(io.BytesIO(response.content))

# We want to animate the driving segment between two track points.
# Everything before the start point will always be marked as an already driven segment
# We ignore flights for now and simply don't draw anything before the end of the last flight segment
# Everything after the end point will be ignored for the entire animation
def animate_driving_segment(route, flight_segments, start_i, end_i, output_path):
    # Find the bounding box of the animation
    bbox = get_animation_bbox(route[start_i:end_i + 1])
    size = (1920, 1080)
    static_map = fetch_map(bbox, size)
    static_map.create_route_animation(route, (start_i, end_i), flight_segments, output_path)

# For now we don't draw driving segments into flight path animations because most of it would just
# be a big blue blob
def animate_flight_segent(start, end, output_path):
    # Find the bounding box of the animation
    bbox = get_animation_bbox([start, end])
    size = (1920, 1080)
    static_map = fetch_map(bbox, size)
    static_map.create_flight_animation(start, end, output_path)

def get_gps_coords(image):
    # Open the image file
    
    # Extract EXIF data
    exif_data = image._getexif()

    if not exif_data:
        return None
    
    # Extract GPS info
    gps_info = {}
    for tag, value in exif_data.items():
        tag_name = PIL.ExifTags.TAGS.get(tag)
        if tag_name == "GPSInfo":
            for key in value:
                gps_tag_name = PIL.ExifTags.GPSTAGS.get(key)
                gps_info[gps_tag_name] = value[key]
    
    # Extract the GPS coordinates
    if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
        latitude = gps_info['GPSLatitude']
        latitude_ref = gps_info['GPSLatitudeRef']
        longitude = gps_info['GPSLongitude']
        longitude_ref = gps_info['GPSLongitudeRef']

        # Convert to degrees
        lat = convert_to_degrees(latitude)
        if latitude_ref != "N":
            lat = -lat

        lon = convert_to_degrees(longitude)
        if longitude_ref != "E":
            lon = -lon

        return lat, lon
    else:
        return None

def convert_to_degrees(value):
    """Helper function to convert the GPS coordinates stored in the EXIF to degrees."""
    d = float(value[0].numerator) / float(value[0].denominator)
    m = float(value[1].numerator) / float(value[1].denominator)
    s = float(value[2].numerator) / float(value[2].denominator)
    return d + (m / 60.0) + (s / 3600.0)

# Main function
def main(gpx_file_path, snapped_points_file, google_maps_api_key, output_directory):
    gmaps = googlemaps.Client(key=google_maps_api_key)
    gpx = parse_gpx(gpx_file_path)
    gpx = fix_gpx_times(gpx)
    waypoints = gpx.waypoints
    track_points = [point for track in gpx.tracks for segment in track.segments for point in segment.points]

    if settings.cache and os.path.exists(snapped_points_file):
        # Load snapped points from file
        print("Loading snapped points from file...")
        snapped_route = load_snapped_points(snapped_points_file)
    else:
        # Snap the GPX track points to the nearest road
        print("Snapping points to road...")
        snapped_route = snap_to_road(gmaps, track_points)
        # Save snapped points to file
        if settings.cache:
            save_snapped_points(snapped_route, snapped_points_file)

    # Debugging output:
    # print all missing originalIndexes
    # last = None
    # total = 0
    # total_dropped_snapped_points = 0
    # total_since_last=0
    # for lat, lon, originalIndex in snapped_route:
    #     if originalIndex is not None and last is not None:
    #         if originalIndex - last > 1:
    #             total += originalIndex - last - 1
    #             total_dropped_snapped_points += total_since_last
    #             total_since_last = 0
    #             print(f"Missing originalIndex between {last} and {originalIndex}")
    #         else:
    #             total_since_last = 0
    #     elif originalIndex is None:
    #         total_since_last += 1
    #     last = originalIndex

    # print(f"Total missing originalIndexes: {total}")
    # print(f"Total dropped snapped points: {total_dropped_snapped_points}")

    # Postprocess the snapped route to create the actual route
    if not settings.cache or not os.path.exists(os.path.join(output_directory, 'final.gpx')):
        route = postprocess_snapped_route(snapped_route, gpx)

        if settings.cache:
            # Save the postprocessed route to a gpx file
            snapped_gpx = gpxpy.gpx.GPX()
            track = gpxpy.gpx.GPXTrack()
            segment = gpxpy.gpx.GPXTrackSegment()
            for point in route:
                segment.points.append(point)
            track.segments.append(segment)
            snapped_gpx.tracks.append(track)
            snapped_gpx.waypoints = gpx.waypoints
            snapped_gpx_file_path = os.path.join(output_directory, 'final.gpx')
            with open(snapped_gpx_file_path, 'w') as f:
                f.write(snapped_gpx.to_xml())
    else:
        print("Loading postprocessed route from file...")
        snapped_gpx_file_path = os.path.join(output_directory, 'final.gpx')
        with open(snapped_gpx_file_path, 'r') as f:
            snapped_gpx = gpxpy.parse(f)
        route = [point for track in snapped_gpx.tracks for segment in track.segments for point in segment.points]

    print(f"Route length: {len(route)} points")
    print(f"Original route length: {len(track_points)} points")
    print(f"Snapped route length: {len(snapped_route)} points")


    if not os.path.exists(os.path.join(output_directory, 'driving_segments.json')) \
        or not os.path.exists(os.path.join(output_directory, 'flight_segments.json')):
        flight_segments, driving_segments = select_animation_segments(route, waypoints)
        # Save driving segments and flight segments to files
        with open(os.path.join(output_directory, 'driving_segments.json'), 'w') as f:
            json.dump(driving_segments, f)
        with open(os.path.join(output_directory, 'flight_segments.json'), 'w') as f:
            json.dump(flight_segments, f)
    else:
        with open(os.path.join(output_directory, 'driving_segments.json'), 'r') as f:
            driving_segments = json.load(f)
        with open(os.path.join(output_directory, 'flight_segments.json'), 'r') as f:
            flight_segments = json.load(f)

    os.makedirs(os.path.join(output_directory, 'anim'), exist_ok=True)

    driving_segments = list(filter(lambda x: (x[1] - x[0] > 0), driving_segments))

    # For a first test, create a single flight animation and a single driving animation
    for driving_segment in driving_segments:
        if driving_segment[1] - driving_segment[0] < 2:
            continue
        print(f"Driving segment from {driving_segment[0]} to {driving_segment[1]}")
        if os.path.exists(os.path.join(output_directory, F"anim/driving_{driving_segment[0]}_{driving_segment[1]}.mp4")):
            print("Driving animation already exists, skipping")
        else:
            animate_driving_segment(route, flight_segments, driving_segment[0], driving_segment[1], os.path.join(output_directory, F"anim/driving_{driving_segment[0]}_{driving_segment[1]}.mp4"))

    for flight_segment in flight_segments:
        print(f"Flight segment from {flight_segment[0]} to {flight_segment[1]}")
        if os.path.exists(os.path.join(output_directory, F"anim/flight_{flight_segment[0]}.mp4")):
            print("Flight animation already exists, skipping")
        else:
            animate_flight_segent(route[flight_segment[0]], route[flight_segment[1]], os.path.join(output_directory, F"anim/flight_{flight_segment[0]}.mp4"))    


    segments = []
    for driving_segment in driving_segments:
        segments.append((driving_segment[0], driving_segment[1], 'road'))
    
    for flight_segment in flight_segments:
        segments.append((flight_segment[0], flight_segment[1], 'flight'))

    segments.sort(key=lambda x: x[0])

    # Create a new ODP document
    # We must describe the dimensions of the page
    doc = odf.opendocument.OpenDocumentPresentation()

    pagelayout = odf.style.PageLayout(name="MyLayout")
    doc.automaticstyles.addElement(pagelayout)
    pagelayout.addElement(
        odf.style.PageLayoutProperties(margin="0pt", pagewidth="1440pt",
                                        pageheight="810pt", printorientation="landscape"))

    # Style for the title frame of the page
    # We set a centered 34pt font with yellowish background
    titlestyle = odf.style.Style(name="MyMaster-title", family="presentation")
    titlestyle.addElement(odf.style.ParagraphProperties(textalign="center"))
    titlestyle.addElement(odf.style.TextProperties(fontsize="34pt"))
    titlestyle.addElement(odf.style.GraphicProperties(fillcolor="#ffff99"))
    doc.styles.addElement(titlestyle)

    # Style for the photo frame
    photostyle = odf.style.Style(name="MyMaster-photo", family="presentation")
    photostyle.addElement(odf.style.ParagraphProperties(textalign="center"))
    photostyle.addElement(odf.style.GraphicProperties(fillcolor="#000000"))
    doc.styles.addElement(photostyle)

    # Create automatic transition
    dpstyle = odf.style.Style(name="dp1", family="drawing-page")
    doc.automaticstyles.addElement(dpstyle)

    # Every drawing page must have a master page assigned to it.
    masterpage = odf.style.MasterPage(name="MyMaster", pagelayoutname=pagelayout)
    doc.masterstyles.addElement(masterpage)

    timezones = os.listdir(settings.photos_dir)
    timezones_images = {}
    for base in timezones:
        images = os.listdir(os.path.join(settings.photos_dir, base))
        images = [os.path.join(settings.photos_dir, base, image) for image in images if image.endswith('.jpg') or image.endswith('.jpeg') or image.endswith('.JPG')]
        timezones_images[base.replace("_", "/")] = images
    
    # sort images by their exif timestamp
    def get_exif_timestamp(img_path):
        img = PIL.Image.open(img_path)
        # Extract EXIF data
        exif_data = img._getexif()

        original = None

        # Check if EXIF data exists
        if exif_data:
            for tag, value in exif_data.items():
                tag_name = PIL.ExifTags.TAGS.get(tag, tag)
                if tag_name == "GPSInfo":
                    gps_time = None
                    gps_date = None
                    for key in value:
                        gps_tag_name = PIL.ExifTags.GPSTAGS.get(key)
                        raw = value[key]
                        if gps_tag_name == "GPSTimeStamp":
                            gps_time = datetime.time(int(raw[0]), int(raw[1]), int(raw[2]))
                        if gps_tag_name == "GPSDateStamp":
                            gps_date = datetime.datetime.strptime(raw, "%Y:%m:%d")
                    if gps_time and gps_date:
                        dt = datetime.datetime.combine(gps_date, gps_time, timezone("UTC"))
                        # Avoid can't compare offset-naive and offset-aware datetimes
                        # Get timezone from exif data
                        return dt
                elif tag_name == "DateTimeOriginal":
                    dt = datetime.datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                    # if the photo has no timezone, assume it was taken in the pacific timezone
                    original = dt

        if original:
            return original

        # If no DateTimeOriginal tag found
        return datetime.datetime.fromtimestamp(os.path.getmtime(img_path)).replace(tzinfo=timezone("US/Pacific"))
    
    # add timestamp to images
    #images = [(img, get_exif_timestamp(os.path.join(settings.photos_dir, img))) for img in images]
    images = []
    tf = timezonefinder.TimezoneFinder()
    for tz_string, images_list in timezones_images.items():
        for img in images_list:
            timestamp = get_exif_timestamp(img)
            tz = timezone(tz_string)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=tz)
            
            next_route_point = min(route, key=lambda x: abs((x.time - timestamp).total_seconds()))
            # Extract timezone from GPX point
            local_tz = tf.timezone_at(lat=next_route_point.latitude, lng=next_route_point.longitude)
            local_tz = timezone(local_tz)
            
            timestamp = timestamp.astimezone(local_tz)
            images.append((img, timestamp))

    # For debugging you can restrict it to a subset of images here for it to run faster
    # images = images[0:25]
    
    images.sort(key=lambda x: x[1])

    images_minimaps = []

    os.makedirs(os.path.join(output_directory, 'minimaps'), exist_ok=True)

    for i in range(len(images)):
        img = Image.open(images[i][0])

        print(f"Creating minimap {i}/{len(images)}", end='\r')

        # check whether we already have one saved
        img_name = os.path.basename(images[i][0])
        if os.path.exists(os.path.join(output_directory, f"minimaps/{img_name}.png")):
            images_minimaps.append(Image.open(os.path.join(output_directory, f"minimaps/{img_name}.png")))
            continue

        # If the image has a gps tag, we take the gps tag, otherwise we take the 
        # temporally closest route point if there is one within 15 minutes
        # or if the next points forward and backwards are within 500m of each other
        gps = get_gps_coords(img)
        minimap = None
        if gps:
            minimap = get_minimap(gps)
        else:
            timestamp = images[i][1]
            closest_point = min(route, key=lambda x: abs((x.time - timestamp).total_seconds()))
            if abs((closest_point.time - timestamp).total_seconds()) < 900:
                minimap = get_minimap((closest_point.latitude, closest_point.longitude))
            else:
                route_past = list(filter(lambda x: x.time < timestamp, route))
                route_future = list(filter(lambda x: x.time > timestamp, route))
                if len(route_past) > 0 and len(route_future) > 0:
                    closest_past = min(route_past, key=lambda x: abs((x.time - timestamp).total_seconds()))
                    closest_future = min(route_future, key=lambda x: abs((x.time - timestamp).total_seconds()))
                    if calculate_distance(closest_past, closest_future) < 500:
                        minimap = get_minimap((closest_past.latitude, closest_past.longitude))
        images_minimaps.append(minimap)
        if minimap is not None:
            minimap.save(os.path.join(output_directory, f"minimaps/{img_name}.png"))

    print("Creating minimaps done                                               ")

    used_images = set()

    # Generate and add slides for each segment
    for i, segment in enumerate(segments):
        segment_cut_timestamp = route[segment[0]].time + (route[segment[1]].time - route[segment[0]].time) * 0.5
        # insert all the photos that happened between the last segment and this one
        for j, (img, timestamp) in enumerate(images):
            if timestamp < segment_cut_timestamp and img not in used_images:
                add_photo_slide(doc, img, dpstyle, masterpage, titlestyle, photostyle, images_minimaps[j], timestamp)
                used_images.add(img)
                print(f"Creating presentation {len(used_images)}/{len(images)}", end='\r')
        start, end, seg_type = segment
        if seg_type == 'road':
            video_path = os.path.join(output_directory, f'anim/driving_{start}_{end}.mp4')
            add_video_slide(doc, video_path, dpstyle, masterpage, titlestyle, photostyle)
        elif seg_type == 'flight':
            video_path = os.path.join(output_directory, f'anim/flight_{start}.mp4')
            add_video_slide(doc, video_path, dpstyle, masterpage, titlestyle, photostyle)

    # insert all the photos that happened after the last segment
    for j, (img, timestamp) in enumerate(images):
        if img not in used_images:
            add_photo_slide(doc, img, dpstyle, masterpage, titlestyle, photostyle, images_minimaps[j], timestamp)
            used_images.add(img)
            print(f"Creating presentation {len(used_images)}/{len(images)}", end='\r')
    
    print("Creating presentation done                                               ")

    # Save the ODP document
    odp_output_path = 'output.odp'
    doc.save(odp_output_path)
    print(f"Presentation saved at {odp_output_path}")

# Usage
if __name__ == "__main__":

    global google_maps_api_key
    global output_directory

    load_dotenv()

    
    main(settings.gpx_file_path, settings.snapped_points_file, settings.google_maps_api_key, settings.output_directory)
