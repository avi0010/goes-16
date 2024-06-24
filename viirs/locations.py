import os, json, argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--points-file", help="Path to geojson file that contains the points for detection", required=True)

    args = parser.parse_args()

    json_path = args.points_file

    geojson = json.load(open(json_path, 'r'))

    locations = []
    ID_SUFFIX = 'US12JUN'

    for index, feature in enumerate(geojson['features']):
        lon, lat = feature['geometry']['coordinates']
        id = ID_SUFFIX + str(index)

        locations.append({
            "_id": id,
            "DeviceId": id,
            "Device": id, 
            "DeviceType": "Incident",
            "lon": lon,
            "lat": lat
        })
    
    # saving file to same dir as that of given json file
    save_dir = os.path.dirname(json_path)
    with open(os.path.join(save_dir, 'locations.json'), 'w') as f:
        json.dump(locations, f, indent=2)