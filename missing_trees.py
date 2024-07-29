# LOAD PACKAGES
from flask import Flask, request, jsonify
from shapely.geometry import Polygon, Point
from scipy.stats import gaussian_kde
from scipy.ndimage import minimum_filter
import numpy as np
import requests
import json
import os

# INITIATE APP
app = Flask(__name__)


# PREDEFINE FUNCTIONS
def pull_orchard_polygon(orchard_id):

    "Pull polygon data for orchard_id 216269"

    id = orchard_id

    url = 'https://api.aerobotics.com/farming/orchards/%s/' % id

    headers = {
        "accept": "application/json",
        "Authorization": "Bearer {insert token}"
    }

    poly_response = requests.get(url, headers=headers)
    data = json.loads(poly_response.text)
    polygon_coordinates = data['polygon']

    poly_coords_list = polygon_coordinates.split(' ')

    # poly coords in incorrect order so switch them
    poly_coords = [(float(coord.split(',')[1]), float(coord.split(',')[0])) for coord in poly_coords_list]

    return poly_coords


def pull_tree_data(orchard_id):

    "Pull coordinates for trees within orchard/polygon"

    id = orchard_id

    url = "https://api.aerobotics.com/farming/surveys/?orchard_id=%s" % id

    headers = {
        "accept": "application/json",
        "Authorization": "Bearer {insert token}"
    }

    survey_response = requests.get(url, headers=headers)
    survey = json.loads(survey_response.text)
    survey_id = survey['results'][0]['id']

    url = "https://api.aerobotics.com/farming/surveys/%s/tree_surveys/" % survey_id

    headers = {
        "accept": "application/json",
        "Authorization": "Bearer 5d03db72854d43a8ce0c63e0d4fb4a261bc29b95ea46b541f537dbf0891b45d6"
    }

    tree_response = requests.get(url, headers=headers)

    tree_data = json.loads(tree_response.text)['results']

    return tree_data


def extract_tree_coords(tree_data):

    tree_coords = [(tree['lat'], tree['lng']) for tree in tree_data]

    return tree_coords


def extract_tree_area(tree_data):
  
    tree_area = [(tree['area']) for tree in tree_data]

    return tree_area


def extract_tree_ndre(tree_data):

    tree_ndre = [(tree['ndre']) for tree in tree_data]

    return tree_ndre


def find_missing_trees_kde(poly_coords, tree_coords, tree_area, num_points, bwe, threshold_percentile, inner_buffer):

    # poly
    poly = Polygon(poly_coords)
    minx, miny, maxx, maxy = poly.bounds

    # inner polygon with buffer
    inner_poly = poly.buffer(-inner_buffer)

    # grid
    x = np.linspace(minx, maxx, num_points)
    y = np.linspace(miny, maxy, num_points)
    xx, yy = np.meshgrid(x, y)

    # kde
    tree_coords_array = np.array(tree_coords)
    tree_area_array = np.array(tree_area)
    kde = gaussian_kde(tree_coords_array.T, weights=tree_area_array, bw_method=bwe)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    density = np.reshape(kde(positions).T, xx.shape)

    # find density only for points in poly
    in_inner_poly = np.array([inner_poly.contains(Point(x, y)) for x, y in zip(xx.ravel(), yy.ravel())])
    density[~in_inner_poly.reshape(xx.shape)] = np.nan

    # find local minima
    neighborhood_size = 10
    local_min = minimum_filter(density, size=neighborhood_size, mode='constant', cval=np.inf)
    min_mask = (density == local_min)

    # use threshold on minima
    threshold = np.nanpercentile(density, threshold_percentile)
    significant_minima = min_mask & (density < threshold)

    # find coords for minima
    potential_missing_trees = [(xx[significant_minima][i], yy[significant_minima][i]) for i in range(np.sum(significant_minima))]

    output_format = {"missing_trees": [{"lat": round(lat, 6), "lng": round(lng, 6)} for lat, lng in potential_missing_trees]}

    return output_format


def find_unhealthy_trees_std(tree_data, tree_ndre):
   
    "Locate potential unhealthy trees for orchard 216269"

    mean = np.mean(tree_ndre)
    std_dev = np.std(tree_ndre)

    tree_unhealthy = [(tree['lat'], tree['lng']) for tree in tree_data if tree['ndre'] < mean - (2*std_dev)]

    output_format = {"unhealthy_trees": [{"lat": round(lat, 6), "lng": round(lng, 6)} for lat, lng in tree_unhealthy]}

    return output_format


# PULL TREE DATA
tree_data = pull_tree_data(216269)


# CREATE END-POINTS

@app.route('/detect_missing_trees', methods=['GET'])
def detect_missing_trees():
  
    poly_coords = pull_orchard_polygon(216269)
    tree_coords = extract_tree_coords(tree_data)
    tree_area = extract_tree_ndre(tree_data)

    potential_missing_trees = find_missing_trees_kde(
      poly_coords,
      tree_coords,
      tree_area,
      num_points=200, 
      bwe=0.007, 
      threshold_percentile=0.12, 
      inner_buffer=0.00008
    )

    return potential_missing_trees

@app.route('/detect_unhealthy_trees', methods=['GET'])
def detect_unhealthy_trees():
   
    tree_ndre = extract_tree_ndre(tree_data)
    potential_unhealthy_trees = find_unhealthy_trees_std(tree_data, tree_ndre)

    return potential_unhealthy_trees


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)