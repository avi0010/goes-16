import os
import pandas as pd
import numpy as np

csv_file = './files/bck/WFIGS_Interagency_Perimeters_YearToDate_6406632943753262912.csv'

df = pd.read_csv(csv_file)

small_fires = 0
medium_fires = 0
total_fires = 0

# Removing rows with null start or end date
df = df.dropna(subset=['poly_PolygonDateTime'],axis=0)
df = df.dropna(subset=['attr_ContainmentDateTime'],axis=0)

for _, row in df.iterrows():
    area = row['poly_GISAcres']
    start_date = row['poly_PolygonDateTime']
    end_date = row['attr_ContainmentDateTime']

    if area < 30:
        continue

    if area <= 100 and area >= 30:
        small_fires += 1

    if area >= 100 and area <= 500:
        medium_fires += 1

    total_fires += 1

print(f"Out of {total_fires} fires, small fires are {small_fires} and medium fires are {medium_fires}")