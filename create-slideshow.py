import datetime
import argparse
import sys
import os
import json
import io
import requests
import math
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import PIL.ImageOps
import PIL.ExifTags
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

import odf
import odf.namespaces
import odf.opendocument
import odf.style
import odf.draw

import timezonefinder
import gpxpy
from matplotlib import font_manager
import numpy as np

# Try to register pillow_heif for HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

import settings
from gmaps_util import (
    fetch_map, get_minimap, load_snapped_points,
    postprocess_snapped_route, save_snapped_points, snap_to_road,
    get_animation_bbox
)
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
                        if total_distance > 0:
                            distance = calculate_distance(segment.points[i], segment.points[k])
                            time_fraction = distance / total_distance
                        else:
                            time_fraction = (k - i) / (j - i)
                        segment.points[k].time = segment.points[i].time + total_time * time_fraction
                i = j
    return gpx


# Function to add a video slide to the ODP presentation
def add_video_slide(doc: odf.opendocument.OpenDocumentPresentation, video_path, dpstyle, masterpage, titlestyle, videostyle):
    ext = os.path.splitext(video_path)[1].lower()
    mimetypes = {
        '.mp4': 'video/mp4',
        '.mov': 'video/quicktime',
        '.avi': 'video/x-msvideo',
        '.mkv': 'video/x-matroska',
    }
    mime_type = mimetypes.get(ext, 'video/mp4')

    page = odf.draw.Page(stylename=dpstyle, masterpagename=masterpage)
    doc.presentation.addElement(page)

    p = doc.addPicture(video_path, mediatype=mime_type)

    videoframe = odf.draw.Frame(stylename=videostyle, width="1440pt", height="810pt", x="0pt", y="0pt")
    page.addElement(videoframe)
    video = odf.draw.Plugin(href=p)
    video.setAttrNS(odf.namespaces.XLINKNS, 'href', p)
    video.setAttrNS(odf.namespaces.XLINKNS, 'type', 'simple')
    video.setAttrNS(odf.namespaces.XLINKNS, 'show', 'embed')
    video.setAttrNS(odf.namespaces.XLINKNS, 'actuate', 'onLoad')
    video.setAttrNS(odf.namespaces.DRAWNS, 'mime-type', mime_type)
    videoframe.addElement(video)


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
    image.thumbnail(target_size, PIL.Image.LANCZOS)

    # Create a new image with the target size and black background
    new_image = PIL.Image.new('RGB', target_size, (0, 0, 0))

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
    try:
        with open(photo_path, 'rb') as f:
            pictdata = f.read()
        img, ct, orig_w, orig_h = getImageInfo(pictdata) # Get dimensions in pixels
    except Exception as e:
        print(f"Warning: Failed to load image '{photo_path}': {e}")
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

    page = odf.draw.Page(stylename=dpstyle, masterpagename=masterpage)
    doc.presentation.addElement(page)
    photoframe = odf.draw.Frame(stylename=photostyle, width="%fpt" % pres_w_pt, height="%fpt" % pres_h_pt, x="%fpt" % 0, y="%fpt" % 0)
    page.addElement(photoframe)
    
    # Scale to the correct size
    img = img.resize((int(w_px), int(h_px)), PIL.Image.Resampling.LANCZOS)
    img = add_black_bars(img, offset_x, offset_y)
    
    # Add the minimap at 50px from the bottom right corner
    minimap_height = 0
    if minimap is not None:
        img.paste(minimap, (1920 - minimap.width, 1080 - minimap.height))
        minimap_height = minimap.height
        
    # Add the timestamp right above the minimap with a height of 50px and a width of minimap.width
    if timestamp is not None:
        draw = PIL.ImageDraw.Draw(img)
        font = font_manager.FontProperties(family='sans-serif', weight='bold')
        file = font_manager.findfont(font)
        font = PIL.ImageFont.truetype(file, 28)
        timestamp_text = timestamp.strftime("%d.%m.%Y %H:%M")
        draw.text((1920 - 10, 1080 - minimap_height - 10), timestamp_text, fill=(255, 255, 255), anchor='rb', align='center', font=font, stroke_fill=(0, 0, 0), stroke_width=2)
        
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='jpeg')
    img_bytes.seek(0)
    img_bytes = img_bytes.read()
    href = doc.addPictureFromString(img_bytes, mediatype='image/jpeg')
    photoframe.addElement(odf.draw.Image(href=href))


def select_animation_segments(route, waypoints):
    flying_segments = []
    driving_segments = []
    flight_cutoff = getattr(settings, 'flight_cutoff_km', 300)
    anim_cutoff = getattr(settings, 'animation_cutoff_km', 25)
    
    # Determine flying segments by looking for consecutive track points with a distance > flight_cutoff
    for i in range(len(route) - 1):
        if calculate_distance(route[i], route[i + 1]) > flight_cutoff:
            flying_segments.append((i, i + 1))
            
    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        end = waypoints[i + 1]
        distance = calculate_distance(start, end)
        if distance < flight_cutoff and distance > anim_cutoff: # Ignore short distances and flights
            # find closest points
            start_i = 0
            start_point = route[0]
            start_distance = calculate_distance(start, start_point)
            end_i = 0
            end_point = route[0]
            first_end_i = None
            first_end_point = None
            end_distance = calculate_distance(end, end_point)
            
            for j in range(1, len(route)):
                start_set = False
                if route[j].time < end.time:
                    start_distance_j = calculate_distance(start, route[j])
                    # We prefer later points even if they are slightly further away
                    # And generally also allow any point that is within 2% of the distance
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
            driving_segments.append((start_i, end_i))

    return flying_segments, driving_segments


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
    def to_float(val):
        if hasattr(val, 'numerator') and hasattr(val, 'denominator'):
            if val.denominator == 0:
                return 0.0
            return float(val.numerator) / float(val.denominator)
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    d = to_float(value[0])
    m = to_float(value[1])
    s = to_float(value[2])
    return d + (m / 60.0) + (s / 3600.0)


def resolve_media_timestamp(file_path, route, tf, default_tz_str="US/Pacific"):
    """
    Extracts or estimates the timestamp for a media file (photo or video),
    returning a timezone-aware datetime.
    """
    is_video = file_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))
    gps_time = None
    gps_date = None
    gps_coords = None
    dt_original = None

    if not is_video:
        try:
            with PIL.Image.open(file_path) as img:
                exif_data = img._getexif()
                if exif_data:
                    gps_coords = get_gps_coords(img)
                    for tag, value in exif_data.items():
                        tag_name = PIL.ExifTags.TAGS.get(tag, tag)
                        if tag_name == "GPSInfo":
                            for key in value:
                                gps_tag_name = PIL.ExifTags.GPSTAGS.get(key)
                                raw = value[key]
                                if gps_tag_name == "GPSTimeStamp":
                                    try:
                                        h = int(float(raw[0]))
                                        m = int(float(raw[1]))
                                        s = int(float(raw[2]))
                                        gps_time = datetime.time(h, m, s)
                                    except Exception:
                                        pass
                                elif gps_tag_name == "GPSDateStamp":
                                    try:
                                        gps_date = datetime.datetime.strptime(raw, "%Y:%m:%d")
                                    except Exception:
                                        pass
                        elif tag_name == "DateTimeOriginal":
                            try:
                                dt_original = datetime.datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                            except Exception:
                                pass
        except Exception as e:
            print(f"Warning: Failed to read EXIF from '{file_path}': {e}")

    # 1. If GPS timestamp is available, it is in UTC.
    if gps_date and gps_time:
        return datetime.datetime.combine(gps_date, gps_time, datetime.timezone.utc)

    # 2. Extract naive local timestamp from EXIF or file modification time.
    if dt_original:
        naive_dt = dt_original
    else:
        # Fallback to mtime
        naive_dt = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))

    # 3. Guess timezone based on the parent folder (e.g. Europe_Berlin -> Europe/Berlin)
    parent_dir = os.path.basename(os.path.dirname(file_path))
    tz_guess = None
    if parent_dir:
        tz_str = parent_dir.replace("_", "/")
        try:
            tz_guess = ZoneInfo(tz_str)
        except Exception:
            pass

    if tz_guess is None:
        try:
            tz_guess = ZoneInfo(default_tz_str)
        except Exception:
            tz_guess = datetime.timezone.utc

    # 4. If we have GPS coordinates, find timezone at those coordinates
    if gps_coords:
        local_tz_str = tf.timezone_at(lat=gps_coords[0], lng=gps_coords[1])
        if local_tz_str:
            try:
                local_tz = ZoneInfo(local_tz_str)
                return naive_dt.replace(tzinfo=local_tz)
            except Exception:
                pass

    # 5. Localize naive timestamp to tz_guess first
    try:
        aware_guess = naive_dt.replace(tzinfo=tz_guess)
    except Exception:
        aware_guess = naive_dt.replace(tzinfo=datetime.timezone.utc)

    # 6. Find temporally closest trackpoint in the GPX route
    if route:
        closest_point = min(route, key=lambda x: abs((x.time - aware_guess).total_seconds()))
        local_tz_str = tf.timezone_at(lat=closest_point.latitude, lng=closest_point.longitude)
        if local_tz_str is not None:
            try:
                local_tz = ZoneInfo(local_tz_str)
                return naive_dt.replace(tzinfo=local_tz)
            except Exception:
                pass
        else:
            # Over ocean or unknown timezone: fallback to folder timezone
            return aware_guess

    return aware_guess


# Main function
def main(gpx_file_path, photos_dir, snapped_points_file, google_maps_api_key, output_directory, odp_output_path, cache, tile_source, dry_run):
    gmaps = None
    if tile_source == 'google':
        if not google_maps_api_key:
            print("Error: Google Maps API key is required when using tile-source 'google'.")
            sys.exit(1)
        gmaps = googlemaps.Client(key=google_maps_api_key)

    # 1. Parse GPX file
    print(f"Parsing GPX file: {gpx_file_path}...")
    gpx = parse_gpx(gpx_file_path)
    gpx = fix_gpx_times(gpx)
    waypoints = gpx.waypoints
    track_points = [point for track in gpx.tracks for segment in track.segments for point in segment.points]

    # Ensure output directory exists if caching or generating files
    os.makedirs(output_directory, exist_ok=True)

    # 2. Scanning media files (recursively)
    print(f"Scanning photos directory: {photos_dir}...")
    valid_image_extensions = ('.jpg', '.jpeg', '.png', '.heic')
    valid_video_extensions = ('.mp4', '.mov', '.avi', '.mkv')
    
    media_files = []
    for root, dirs, files in os.walk(photos_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_image_extensions:
                media_files.append((os.path.join(root, file), 'image'))
            elif ext in valid_video_extensions:
                media_files.append((os.path.join(root, file), 'video'))

    # Initialize TimezoneFinder
    tf = timezonefinder.TimezoneFinder()

    # Resolve timestamps for all media
    print(f"Resolving timestamps for {len(media_files)} media files...")
    media_items = []
    for path, media_type in media_files:
        timestamp = resolve_media_timestamp(path, track_points, tf)
        media_items.append((path, timestamp, media_type))

    # Sort media items by timestamp
    media_items.sort(key=lambda x: x[1])

    # Check for dry run
    if dry_run:
        print("\n=== DRY RUN MODE ===")
        print(f"GPX file: {gpx_file_path}")
        print(f"Route length: {len(track_points)} track points")
        
        flight_segs, driving_segs = select_animation_segments(track_points, waypoints)
        
        print("\n--- Detected Flight Segments ---")
        for i, (start_idx, end_idx) in enumerate(flight_segs):
            start_pt = track_points[start_idx]
            end_pt = track_points[end_idx]
            dist = calculate_distance(start_pt, end_pt)
            print(f"  [{i+1}] Trackpoint {start_idx} to {end_idx} ({dist:.1f} km)")
            print(f"      Time: {start_pt.time} -> {end_pt.time}")
            
        print("\n--- Detected Driving Segments ---")
        for i, (start_idx, end_idx) in enumerate(driving_segs):
            start_pt = track_points[start_idx]
            end_pt = track_points[end_idx]
            dist = calculate_distance(start_pt, end_pt)
            print(f"  [{i+1}] Trackpoint {start_idx} to {end_idx} ({dist:.1f} km)")
            print(f"      Time: {start_pt.time} -> {end_pt.time}")
            
        print(f"\nPhotos Directory: {photos_dir}")
        print(f"Total media files found: {len(media_items)}")
        images_count = sum(1 for m in media_items if m[2] == 'image')
        videos_count = sum(1 for m in media_items if m[2] == 'video')
        print(f"  Images (JPEG, PNG, HEIC): {images_count}")
        print(f"  Videos (MP4, MOV, etc.): {videos_count}")
        
        print("\nAPI Quota Estimate:")
        if tile_source == 'google':
            print("  - Snap to Roads: 1 API call per 100 track points")
            print(f"    (Approx. {math.ceil(len(track_points)/100)} requests)")
            print(f"  - Static Maps: {len(driving_segs) + len(flight_segs)} requests (1 per segment)")
        else:
            print("  - Snap to Roads: (Skipped, OSM tile source doesn't support snapping)")
            print("  - OSM Tiles: Tiles will be fetched from OpenStreetMap (no API key required)")
        print("Dry run finished. No API requests were made and no files were modified.")
        return

    # If not a dry run, perform snapping
    if tile_source == 'google':
        if cache and os.path.exists(snapped_points_file):
            print("Loading snapped points from file...")
            snapped_route = load_snapped_points(snapped_points_file)
        else:
            print("Snapping points to road...")
            snapped_route = snap_to_road(gmaps, track_points)
            if cache:
                save_snapped_points(snapped_route, snapped_points_file)
    else:
        print("Tile source is OSM; skipping snap_to_roads (using original route points)...")
        snapped_route = [(p.latitude, p.longitude, idx) for idx, p in enumerate(track_points)]

    # Postprocess the snapped route
    final_gpx_path = os.path.join(output_directory, 'final.gpx')
    if not cache or not os.path.exists(final_gpx_path):
        route = postprocess_snapped_route(snapped_route, gpx)
        if cache:
            snapped_gpx = gpxpy.gpx.GPX()
            track = gpxpy.gpx.GPXTrack()
            segment = gpxpy.gpx.GPXTrackSegment()
            for point in route:
                segment.points.append(point)
            track.segments.append(segment)
            snapped_gpx.tracks.append(track)
            snapped_gpx.waypoints = gpx.waypoints
            with open(final_gpx_path, 'w') as f:
                f.write(snapped_gpx.to_xml())
    else:
        print("Loading postprocessed route from file...")
        with open(final_gpx_path, 'r') as f:
            snapped_gpx = gpxpy.parse(f)
        route = [point for track in snapped_gpx.tracks for segment in track.segments for point in segment.points]

    print(f"Route length: {len(route)} points")
    print(f"Original route length: {len(track_points)} points")
    print(f"Snapped route length: {len(snapped_route)} points")

    # Select animation segments
    driving_segments_json = os.path.join(output_directory, 'driving_segments.json')
    flight_segments_json = os.path.join(output_directory, 'flight_segments.json')
    
    if not cache or not os.path.exists(driving_segments_json) or not os.path.exists(flight_segments_json):
        flight_segments, driving_segments = select_animation_segments(route, waypoints)
        if cache:
            with open(driving_segments_json, 'w') as f:
                json.dump(driving_segments, f)
            with open(flight_segments_json, 'w') as f:
                json.dump(flight_segments, f)
    else:
        print("Loading animation segments from cache...")
        with open(driving_segments_json, 'r') as f:
            driving_segments = json.load(f)
        with open(flight_segments_json, 'r') as f:
            flight_segments = json.load(f)

    os.makedirs(os.path.join(output_directory, 'anim'), exist_ok=True)
    driving_segments = list(filter(lambda x: (x[1] - x[0] > 0), driving_segments))

    # Generate animations
    for driving_segment in driving_segments:
        if driving_segment[1] - driving_segment[0] < 2:
            continue
        print(f"Driving segment from {driving_segment[0]} to {driving_segment[1]}")
        anim_file = os.path.join(output_directory, f"anim/driving_{driving_segment[0]}_{driving_segment[1]}.mp4")
        if os.path.exists(anim_file):
            print("Driving animation already exists, skipping")
        else:
            animate_driving_segment(route, flight_segments, driving_segment[0], driving_segment[1], anim_file)

    for flight_segment in flight_segments:
        print(f"Flight segment from {flight_segment[0]} to {flight_segment[1]}")
        anim_file = os.path.join(output_directory, f"anim/flight_{flight_segment[0]}.mp4")
        if os.path.exists(anim_file):
            print("Flight animation already exists, skipping")
        else:
            animate_flight_segent(route[flight_segment[0]], route[flight_segment[1]], anim_file)

    segments = []
    for driving_segment in driving_segments:
        segments.append((driving_segment[0], driving_segment[1], 'road'))
    for flight_segment in flight_segments:
        segments.append((flight_segment[0], flight_segment[1], 'flight'))
    segments.sort(key=lambda x: x[0])

    # 3. Create ODP document
    doc = odf.opendocument.OpenDocumentPresentation()

    pagelayout = odf.style.PageLayout(name="MyLayout")
    doc.automaticstyles.addElement(pagelayout)
    pagelayout.addElement(
        odf.style.PageLayoutProperties(margin="0pt", pagewidth="1440pt",
                                        pageheight="810pt", printorientation="landscape"))

    titlestyle = odf.style.Style(name="MyMaster-title", family="presentation")
    titlestyle.addElement(odf.style.ParagraphProperties(textalign="center"))
    titlestyle.addElement(odf.style.TextProperties(fontsize="34pt"))
    titlestyle.addElement(odf.style.GraphicProperties(fillcolor="#ffff99"))
    doc.styles.addElement(titlestyle)

    photostyle = odf.style.Style(name="MyMaster-photo", family="presentation")
    photostyle.addElement(odf.style.ParagraphProperties(textalign="center"))
    photostyle.addElement(odf.style.GraphicProperties(fillcolor="#000000"))
    doc.styles.addElement(photostyle)

    dpstyle = odf.style.Style(name="dp1", family="drawing-page")
    doc.automaticstyles.addElement(dpstyle)

    masterpage = odf.style.MasterPage(name="MyMaster", pagelayoutname=pagelayout)
    doc.masterstyles.addElement(masterpage)

    # Generate and cache minimaps for images
    print("Generating minimaps...")
    os.makedirs(os.path.join(output_directory, 'minimaps'), exist_ok=True)
    media_minimaps = [None] * len(media_items)

    def process_minimap(idx):
        media_path, timestamp, media_type = media_items[idx]
        if media_type == 'video':
            return idx, None

        # check whether we already have one saved (using relative path to support subfolders safely)
        rel_path = os.path.relpath(media_path, photos_dir)
        img_cache_name = rel_path.replace(os.sep, '_')
        cache_path = os.path.join(output_directory, f"minimaps/{img_cache_name}.png")
        
        if os.path.exists(cache_path):
            try:
                img = PIL.Image.open(cache_path)
                img.load()  # Load pixels and close file handle
                return idx, img
            except Exception:
                pass

        # Open image to inspect GPS coords
        try:
            with PIL.Image.open(media_path) as img:
                gps = get_gps_coords(img)
        except Exception:
            gps = None

        minimap = None
        if gps:
            minimap = get_minimap(gps)
        else:
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
        
        if minimap is not None:
            minimap.save(cache_path)
            minimap.load()
            
        return idx, minimap

    # We use multiple workers to fetch tiles and build minimaps in parallel (mostly network bound)
    max_workers = min(16, (os.cpu_count() or 4) * 2)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_minimap, i) for i in range(len(media_items))]
        completed = 0
        for future in as_completed(futures):
            try:
                idx, minimap = future.result()
                media_minimaps[idx] = minimap
            except Exception as e:
                print(f"\nError generating minimap: {e}")
            completed += 1
            print(f"Creating minimap {completed}/{len(media_items)}", end='\r')

    print("Creating minimaps done                                               ")

    used_media = set()

    # Generate and add slides for each segment
    for i, segment in enumerate(segments):
        segment_cut_timestamp = route[segment[0]].time + (route[segment[1]].time - route[segment[0]].time) * 0.5
        # insert all the photos/videos that happened before the last segment and this one
        for j, (media_path, timestamp, media_type) in enumerate(media_items):
            if timestamp < segment_cut_timestamp and media_path not in used_media:
                if media_type == 'image':
                    add_photo_slide(doc, media_path, dpstyle, masterpage, titlestyle, photostyle, media_minimaps[j], timestamp)
                else:
                    add_video_slide(doc, media_path, dpstyle, masterpage, titlestyle, photostyle)
                used_media.add(media_path)
                print(f"Creating presentation {len(used_media)}/{len(media_items)}", end='\r')
                
        start, end, seg_type = segment
        if seg_type == 'road':
            video_path = os.path.join(output_directory, f'anim/driving_{start}_{end}.mp4')
            add_video_slide(doc, video_path, dpstyle, masterpage, titlestyle, photostyle)
        elif seg_type == 'flight':
            video_path = os.path.join(output_directory, f'anim/flight_{start}.mp4')
            add_video_slide(doc, video_path, dpstyle, masterpage, titlestyle, photostyle)

    # insert all the photos/videos that happened after the last segment
    for j, (media_path, timestamp, media_type) in enumerate(media_items):
        if media_path not in used_media:
            if media_type == 'image':
                add_photo_slide(doc, media_path, dpstyle, masterpage, titlestyle, photostyle, media_minimaps[j], timestamp)
            else:
                add_video_slide(doc, media_path, dpstyle, masterpage, titlestyle, photostyle)
            used_media.add(media_path)
            print(f"Creating presentation {len(used_media)}/{len(media_items)}", end='\r')
    
    print("Creating presentation done                                               ")

    # Save the ODP document
    doc.save(odp_output_path)
    print(f"Presentation saved at {odp_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a travel slideshow ODP from photos and GPX.")
    parser.add_argument("--gpx", default=settings.gpx_file_path, help="Path to the input GPX file")
    parser.add_argument("--photos", default=settings.photos_dir, help="Path to the photos directory")
    parser.add_argument("--output", default=settings.odp_output_path, help="Path to the output ODP file")
    parser.add_argument("--cache", action="store_true", default=settings.cache, help="Enable caching of intermediate state")
    parser.add_argument("--no-cache", action="store_false", dest="cache", help="Disable caching of intermediate state")
    parser.add_argument("--tile-source", choices=["google", "osm"], default=settings.tile_source, help="Tile source for maps ('google' or 'osm')")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run to list detected segments and exit")
    args = parser.parse_args()

    # Update settings variables with command-line arguments to make sure other modules see them
    settings.gpx_file_path = args.gpx
    settings.photos_dir = args.photos
    settings.odp_output_path = args.output
    settings.cache = args.cache
    settings.tile_source = args.tile_source

    # Validate inputs
    if not os.path.exists(args.gpx):
        print(f"Error: GPX file '{args.gpx}' does not exist.")
        sys.exit(1)

    if not args.photos or not os.path.exists(args.photos):
        print(f"Error: Photos directory '{args.photos}' does not exist.")
        print("Please specify a valid directory using --photos or set PHOTO_DIR/PHOTOS_DIR environment variables.")
        sys.exit(1)

    if args.tile_source == 'google' and not settings.google_maps_api_key:
        print("Error: Google Maps API key is not set. Please set the GOOGLE_MAPS_API_KEY environment variable or use --tile-source osm.")
        sys.exit(1)

    main(
        gpx_file_path=settings.gpx_file_path,
        photos_dir=settings.photos_dir,
        snapped_points_file=settings.snapped_points_file,
        google_maps_api_key=settings.google_maps_api_key,
        output_directory=settings.output_directory,
        odp_output_path=settings.odp_output_path,
        cache=settings.cache,
        tile_source=settings.tile_source,
        dry_run=args.dry_run
    )
