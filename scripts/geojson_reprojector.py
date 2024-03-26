import json
import pyproj
from shapely import area
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import transform
import math

def reproject_geojson(geojson_path, src_crs, dst_crs, output_path):

    def get_coords_from_polygon(shape):
        coords = set()  

        if shape.geom_type == 'Polygon':
            coords.update(shape.exterior.coords[:-1])
            for linearring in shape.interiors:
                coords.update(linearring.coords[:-1])
        elif shape.geom_type == 'MultiPolygon':
            for polygon in shape.geoms:
                coords.update(get_coords_from_polygon(polygon)) 

        return coords


    """
    Reproject a GeoJSON file from the source CRS to the destination CRS.

    Parameters:
    - geojson_path: Path to the input GeoJSON file.
    - src_crs: Source coordinate reference system (CRS) string.
    - dst_crs: Destination coordinate reference system (CRS) string.
    - output_path: Path to save the reprojected GeoJSON file.

    Returns:
    - None
    """
    # Load GeoJSON data
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)

    # Define projection transformers
    project = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform
    reverse_project = pyproj.Transformer.from_crs(dst_crs, src_crs, always_xy=True).transform

    feature_layer = {
        "type": "FeatureCollection",
        "name": "Perimeters",
        "crs": {
            "type": "name",
            "properties": {
            }
        },
        "features": []
    }

    # Reproject each feature geometry
    for feature in geojson_data['features']:
        geom = shape(feature['geometry'])
        reprojected_geom = transform(project, geom)

        reprojected_feature = {
            'type': 'feature',
            'properties': feature['properties'],
            'geometry': mapping(reprojected_geom)
        }

        geom = shape(reprojected_feature['geometry'])
        
        coordinates = get_coords_from_polygon(geom)
        invalid_coordinates = False
        for coordinate in coordinates:
            if math.isinf(coordinate[0]) or math.isinf(coordinate[1]):
                invalid_coordinates = True
                break
        if invalid_coordinates:
            continue                

        feature_layer['features'].append(reprojected_feature)
        
    print(len(feature_layer['features']))
    # Write reprojected GeoJSON to file
    with open(output_path, 'w') as f:
        json.dump(feature_layer, f)

# Example usage
src_crs = 'EPSG:4326'  # Source CRS
dst_crs = 'ESRI:102498'  # Destination CRS

reproject_geojson(geojson_path='./files/WFIGS_Interagency_Perimeters_6317078473007695789.geojson', src_crs=src_crs, dst_crs=dst_crs, output_path='./files/WFIGS_Interagency_Perimeters.json')