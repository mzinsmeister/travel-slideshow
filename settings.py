from dotenv import load_dotenv
import os


load_dotenv()

# Path to the input GPX file
gpx_file_path = 'data/travel-route.gpx'
# Path to the file containing the snapped points (only used if cache=True)
snapped_points_file = 'data/tmp/snapped_points.json'
# Load Google Maps API from environment variable
google_maps_api_key = os.environ.get('GOOGLE_MAPS_API_KEY')
# Output directory for cache and temporary files
output_directory = 'data/tmp/'
# Path to the directory containing the photos
photos_dir = os.environ.get('PHOTOS_DIR') or os.environ.get('PHOTO_DIR')
# Path to save the final presentation
odp_output_path = 'output.odp'
# Width (and height) of the minimap in the pictures in pixels
minimap_width = 350
# Default tile source: 'google' or 'osm'
tile_source = 'google'
# Cache different forms of intermediate state to speed up debugging/development
cache = False

# Configurable thresholds (in kilometers)
animation_cutoff_km = 25
flight_cutoff_km = 300
snap_break_km = 50