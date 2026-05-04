import pandas as pd
import pickle
import os
import folium
import branca.colormap as cm
from pyproj import Transformer


# read files
INPUT_DIR_MODELRESULTS = "/scratch/mch/tkluser/rainforest_semester_project/saved_models/GRU_Baseline_diverseOptim2/cv_1_GRU_Baseline_diverseOptim2_all_folds.parquet"
OUTPUT_DIR = "/scratch/mch/tkluser/rainforest_semester_project/saved_models/GRU_Baseline_diverseOptim2/station_mse_map.html"

# INPUT_DIR_MODELRESULTS = "/scratch/mch/tkluser/rainforest_semester_project/saved_models/GRU_Baseline_Denseweight_alpha10/cv_1_GRU_Baseline_Denseweight_alpha10_all_folds.parquet"
# OUTPUT_DIR = "/scratch/mch/tkluser/rainforest_semester_project/saved_models/GRU_Baseline_Denseweight_alpha10/station_mse_map.html"

# INPUT_DIR_MODELRESULTS = "/scratch/mch/wolfensb/rainforest_semester_project/saved_models/cv_predictions/cv_pred_RF_all_folds.parquet"
# OUTPUT_DIR = "/scratch/mch/tkluser/rainforest_semester_project/station_mse_map_RF.html"


INPUT_DIR_GAUGEDATA = "/store_new/mch/msrad/radar/radar_database_v2/rf_input_data_qc/"
df_stations = pd.read_csv("/store_new/mch/msrad/radar/rainforest_data/references/metadata/data_stations.csv",
                          encoding='latin1',
                          sep=";")
gauge = pd.read_parquet(os.path.join(INPUT_DIR_GAUGEDATA, "gauge.parquet"))
df_modelresults = pd.read_parquet(INPUT_DIR_MODELRESULTS)
df_modelresults = df_modelresults[df_modelresults["split"] == "test"]

# vertgroup is the gauge index 
df_modelresults = pd.merge(df_modelresults, gauge, left_on="vertgroup", right_index=True)
df_modelresults["squared_error"] = df_modelresults["error"] ** 2
df_modelresults = df_modelresults.groupby("STATION").agg(mse=("squared_error", "mean"))
df_modelresults = pd.merge(df_modelresults, df_stations, left_on="STATION", right_on="Abbrev")

# Setup the transformer
# From LV95 (2056) to WGS84 (4326). 
# Change 2056 to 21781 if you are using the old LV03 system.
transformer = Transformer.from_crs("EPSG:21781", "EPSG:4326", always_xy=True)

# 2. Define a function for the transformation
def convert_coords(row):
    # always_xy=True means we input (Easting, Northing) 
    # and get back (Longitude, Latitude)
    lon, lat = transformer.transform(row['Y'], row['X'])
    return pd.Series([lat, lon])

# Apply to your dataframe
# This creates two new columns: 'lat' and 'lon'
df_modelresults[['lat', 'lon']] = df_modelresults.apply(convert_coords, axis=1)

min_mse = df_modelresults['mse'].min()
max_mse = df_modelresults['mse'].max()

colormap = cm.LinearColormap(
    colors=['blue', 'green', 'yellow', 'orange', 'red'],
    vmin=min_mse,
    vmax=max_mse,
    caption='Mean Squared Error (MSE) per Station'
)


map_center = [df_modelresults['lat'].mean(), df_modelresults['lon'].mean()]
m = folium.Map(location=map_center, zoom_start=8, tiles='cartodbpositron')

for _, row in df_modelresults.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=7,  # Size of the marker
        popup=f"Station: {row['Abbrev']}<br>MSE: {row['mse']:.4f}",
        tooltip=row['Abbrev'],
        color=colormap(row['mse']),  # Border color
        fill=True,
        fill_color=colormap(row['mse']),  # Fill color based on MSE
        fill_opacity=0.7
    ).add_to(m)

colormap.add_to(m)
m.save(OUTPUT_DIR)


