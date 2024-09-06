from dotenv import load_dotenv
import os


load_dotenv()

# Path to the input GPX file
gpx_file_path = 'data/travel-route.gpx'
# Path to the file containing the snapped points (only used if cache=True)
snapped_points_file = 'data/tmp/snapped_points.json'
# Load Google Maps API from environment variable
google_maps_api_key = os.environ.get('GOOGLE_MAPS_API_KEY')
# Load Google Maps API from environment variable
output_directory = 'data/tmp/'
# Path to the directory containing the photos
photos_dir = os.environ.get('PHOTOS_DIR')
# Width (and height) of the minimap in the pictures  in pixels
minimap_width = 350
# Cache different forms of intermediate state to speed up debugging/development
# Be careful if setting this to True as it may require manual deletion of files
# in the output directory if the input or code changes
cache=False