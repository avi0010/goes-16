import os
import pandas as pd
import numpy as np
from datetime import datetime
import csv
import json
import geojson
import fiona

csv_file = './files/bck/WFIGS_Interagency_Perimeters_-3500393626074286023.csv'
out_csv_file = './files/bck/latest_edit_based_analysis_WFIGS_Interagency_Perimeters_-3500393626074286023.csv'

HOUR_LIMIT = 4

csv_out_rows = [['minimum_days', 'small', 'medium', 'large', 'earliest_start_date', 'latest_end_date']]

def monthly_fires_generator():
    monthly_fires = dict()

    df = pd.read_csv(csv_file)

    print(df['poly_MapMethod'].unique())

    # Removing rows with null start or end date or area
    df = df.dropna(subset=['poly_GISAcres','attr_ControlDateTime', 'poly_PolygonDateTime', 'poly_MapMethod'],axis=0)

    for _, row in df.iterrows():
        area = row['poly_GISAcres']
        start_date = datetime.strptime(row['poly_PolygonDateTime'], '%m/%d/%Y %I:%M:%S %p')
        end_date = datetime.strptime(row['attr_ControlDateTime'], '%m/%d/%Y %I:%M:%S %p')   

        large = True
        if area < 30:
            continue

        if start_date.year != 2023:
            continue  

        if row['poly_MapMethod'] in ['IR Image Interpretation', 'Hand Sketch', 'Modeled', 'Infrared Image', 'Auto-generated',
                'Phone/Tablet', 'Other', 'Vector', 'MixedMethods', 'Digitized-Other', 'Digitized', 'Image Interpretation',
                'Digitized-Topo']:
            continue    

        gap = (end_date - start_date)    

        if (gap.days * 24 + gap.seconds // 3600) < HOUR_LIMIT:
            continue   

        month = start_date.month
        if month in monthly_fires:
            if area <= 100 and area >= 30:
                monthly_fires[month]['small_fire'] += 1
            elif area > 100 and area <= 500:
                monthly_fires[month]['medium_fire'] += 1
            else:
                monthly_fires[month]['large_fire'] += 1
            
            if monthly_fires[month]['min_interval'] is None:
                monthly_fires[month]['min_interval'] = gap.days
            else:
                if monthly_fires[month]['min_interval'] > gap.days:
                    monthly_fires[month]['min_interval'] = gap.days

            if monthly_fires[month]['max_interval'] is None:
                monthly_fires[month]['max_interval'] = gap.days
            else:
                if monthly_fires[month]['max_interval'] < gap.days:
                    monthly_fires[month]['max_interval'] = gap.days
        else:
            monthly_fires[month] = {
                'small_fire': 0,
                'medium_fire': 0,
                'large_fire': 0,
                'min_interval': None,
                'max_interval': None
            }
    
    out_csv_path = './files/bck/monthly_WFIGS_Interagency_Perimeters_-3500393626074286023.csv'
    with open(out_csv_path, 'w') as f:
        csv_writer = csv.writer(f)

        rows = [
            ['month', 'small_fires', 'medium_fires', 'large_fires', 'min_interval', 'max_interval']
        ]
        data_rows = [[k, *list(monthly_fires[k].values())] for k in monthly_fires]
        rows.extend(data_rows)
        csv_writer.writerows(rows)

def fire_number_per_day(geojson_path):

    '''df = pd.read_csv(csv_file)

    print(df['poly_MapMethod'].unique())

    # Removing rows with null start or end date or area
    df = df.dropna(subset=['poly_GISAcres','attr_ControlDateTime', 'poly_PolygonDateTime', 'poly_MapMethod'],axis=0)'''


    # Load GeoJSON data
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)

    for day_limit in range(1, 3):

        small_fires = 0
        medium_fires = 0
        total_fires = 0
        largest_area = 0
        earliest_start_date = None
        latest_end_date = None

        for feature in geojson_data['features']:
            row = feature['properties']

            area = row['poly_GISAcres']

            start_date_str = row['poly_PolygonDateTime']
            try:
                start_date = datetime.strptime(start_date_str[start_date_str.find(',')+2::], '%m/%d/%Y %I:%M:%S %p')
            except ValueError as e:
                try:
                    start_date = datetime.strptime(start_date_str[start_date_str.find(',')+2::], '%d %b %Y %H:%M:%S GMT')
                except ValueError as e:
                    print(e)
                    continue

            end_date_str = row['attr_ControlDateTime']
            try:
                end_date = datetime.strptime(end_date_str[end_date_str.find(',')+2::], '%m/%d/%Y %I:%M:%S %p')   
            except ValueError as e:
                try:
                    end_date = datetime.strptime(end_date_str[end_date_str.find(',')+2::], '%d %b %Y %H:%M:%S GMT')
                except ValueError as e:
                    print(e)
                    continue


            large = True
            if area < 30:
                continue

            if start_date.year == 2023:
                pass 
            else:
                continue

            if row['poly_MapMethod'] in ['IR Image Interpretation', 'Hand Sketch', 'Modeled', 'Infrared Image', 'Auto-generated',
                    'Phone/Tablet', 'Other', 'Vector', 'MixedMethods', 'Digitized-Other', 'Digitized', 'Image Interpretation',
                    'Digitized-Topo']:
                pass  

            gap = (end_date - start_date)    
            if start_date > end_date:
                #print(f"Start date greater than end date- {start_date} {end_date}")
                continue

            if (gap.days * 24 + gap.seconds // 3600) < HOUR_LIMIT:
                continue

            if gap.days < day_limit:
                continue

            if area <= 100 and area >= 30:
                small_fires += 1
            elif area > 100 and area <= 500:
                medium_fires += 1
            else:
                if largest_area <= area:
                    largest_area = area
            #print(row['poly_MapMethod'])
            total_fires += 1  

            if earliest_start_date is None:
                earliest_start_date = start_date
            else:
                if earliest_start_date > start_date:
                    earliest_start_date = start_date 

            if latest_end_date is None:
                latest_end_date = end_date
            else:
                if latest_end_date < end_date:
                    latest_end_date = end_date

        csv_out_rows.append([day_limit, small_fires, medium_fires, total_fires - (medium_fires + small_fires), earliest_start_date, latest_end_date])
        print(f"For at least days {day_limit}, out of {total_fires} fires, small fires are {small_fires} and medium fires are {medium_fires} and large fires are {total_fires - (medium_fires + small_fires)}")
        print(f"Largest area- {largest_area}, earliest start date- {earliest_start_date}, latest end date- {latest_end_date}")


    with open(out_csv_file, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(csv_out_rows)

def filter_perimeters(geojson_path:str, output_path, day_limit:int=1):

    # Load GeoJSON data
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)

    featurecollection_layer = {
        "type": "FeatureCollection",
        "name": "Perimeters",
        "crs": {
            "type": "name",
            "properties": {
            }
        },
        "features": []
    }

    features = []


    small_fires = 0
    medium_fires = 0
    total_fires = 0
    largest_area = 0
    earliest_start_date = None
    latest_end_date = None

    start_end_date_anomaly = 0
    fire_count_2023 = 0
    less_reliable_methods = 0

    print(len(geojson_data['features']))
    for feature in geojson_data['features']:
        properties = feature['properties']

        area = properties['poly_GISAcres']
        start_date_str = properties['poly_PolygonDateTime']
        try:
            start_date = datetime.strptime(start_date_str[start_date_str.find(',')+2::], '%m/%d/%Y %I:%M:%S %p')
        except ValueError as e:
            try:
                start_date = datetime.strptime(start_date_str[start_date_str.find(',')+2::], '%d %b %Y %H:%M:%S GMT')
            except ValueError as e:
                print(e)
                continue

        end_date_str = properties['attr_FireOutDateTime']
        try:
            end_date = datetime.strptime(end_date_str[end_date_str.find(',')+2::], '%m/%d/%Y %I:%M:%S %p')   
        except ValueError as e:
            try:
                end_date = datetime.strptime(end_date_str[end_date_str.find(',')+2::], '%d %b %Y %H:%M:%S GMT')
            except ValueError as e:
                print(e)
                continue

        if area < 30:
            continue

        if start_date.year == 2023:
            fire_count_2023 += 1  
        else:
            continue

        if properties['poly_MapMethod'] in ['IR Image Interpretation', 'Hand Sketch', 'Modeled', 'Infrared Image', 'Auto-generated',
                'Phone/Tablet', 'Other', 'Vector', 'MixedMethods', 'Digitized-Other', 'Digitized', 'Image Interpretation',
                'Digitized-Topo']:
            less_reliable_methods += 1
            #continue    

        gap = (end_date - start_date)    
        if start_date > end_date:
            #print(f"Start date greater than end date- {start_date} {end_date}")
            start_end_date_anomaly += 1
            continue

        if (gap.days * 24 + gap.seconds // 3600) < HOUR_LIMIT:
            continue

        if gap.days < day_limit:
            continue

        if area <= 100 and area >= 30:
            small_fires += 1
        elif area > 100 and area <= 500:
            medium_fires += 1
        else:
            if largest_area <= area:
                largest_area = area
        #print(row['poly_MapMethod'])
        total_fires += 1  

        if earliest_start_date is None:
            earliest_start_date = start_date
        else:
            if earliest_start_date > start_date:
                earliest_start_date = start_date 

        if latest_end_date is None:
            latest_end_date = end_date
        else:
            if latest_end_date < end_date:
                latest_end_date = end_date
        
        features.append(feature)
    print(len(features), f'  2023 fires {fire_count_2023} start-end-date-anamoly {start_end_date_anomaly} less reliable methods {less_reliable_methods}')

    print(f"For at least days {day_limit}, out of {total_fires} fires, small fires are {small_fires} and medium fires are {medium_fires} and large fires are {total_fires - (medium_fires + small_fires)}")
    print(f"Largest area- {largest_area}, earliest start date- {earliest_start_date}, latest end date- {latest_end_date}")


    featurecollection_layer['features'] = features
    # Write reprojected GeoJSON to file
    with open(output_path, 'w') as f:
        json.dump(featurecollection_layer, f)

if __name__ == '__main__':
    #monthly_fires_generator()
    filter_perimeters('/home/ubuntu/dev/goes-16/files/WFIGS_Interagency_Perimeters.json', './files/Filtered_WFIGS_Interagency_Perimeters.json', 1)
    #fire_number_per_day('./files/Filtered_WFIGS_Interagency_Perimeters.json')