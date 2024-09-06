from geopy.distance import geodesic
import math

# Function to calculate the km distance between two geographical points, accepts points as dicts with 'latitude' and 'longitude' keys
# or as objects with 'latitude' and 'longitude' attributes or as tuples with latitude and longitude
def calculate_distance(point1, point2) -> float:
    """Calculate the distance between two geographical points."""
    if isinstance(point1, dict):
        lat1, lon1 = point1['latitude'], point1['longitude']
    elif hasattr(point1, 'latitude'):
        lat1, lon1 = point1.latitude, point1.longitude
    else:
        lat1, lon1 = point1

    if isinstance(point2, dict):
        lat2, lon2 = point2['latitude'], point2['longitude']
    elif hasattr(point2, 'latitude'):
        lat2, lon2 = point2.latitude, point2.longitude
    else:
        lat2, lon2 = point2

    return geodesic((lat1, lon1), (lat2, lon2)).kilometers

# 
# MProjection.prototype.fromLatLngToPoint = function(latlng) {
#     var x = (latlng.lng() + 180) / 360 * 256;
#     var y = ((1 - Math.log(Math.tan(latlng.lat() * Math.PI / 180) + 1 / Math.cos(latlng.lat() * Math.PI / 180)) / Math.PI) / 2 * Math.pow(2, 0)) * 256;
#     return new google.maps.Point(x, y);
# };
# MProjection.prototype.fromPointToLatLng = function(point) {
#     var lng = point.x / 256 * 360 - 180;
#     var n = Math.PI - 2 * Math.PI * point.y / 256;
#     var lat = (180 / Math.PI * Math.atan(0.5 * (Math.exp(n) - Math.exp(-n))));
#     return new google.maps.LatLng(lat, lng);
# };

# Convert latitude and longitude google maps world coordinates
def lat_lng_to_world_coords(lat: float, lng: float) -> tuple[float, float]:
    """Convert latitude and longitude to world coordinates."""
    x = (lng + 180) / 360 * 256
    y = ((1 - math.log(math.tan(lat * math.pi / 180) + 1 / math.cos(lat * math.pi / 180)) / math.pi) / 2 * math.pow(2, 0)) * 256
    return (x, y)

# Convert world coordinates to pixel coordinates
def world_coords_to_lat_lng(x: float, y: float) -> tuple[float, float]:
    """Convert world coordinates to latitude and longitude."""
    lng = x / 256 * 360 - 180
    n = math.pi - 2 * math.pi * y / 256
    lat = (180 / math.pi * math.atan(0.5 * (math.exp(n) - math.exp(-n))))
    return (lat, lng)