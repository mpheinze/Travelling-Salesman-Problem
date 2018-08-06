##### ----- //////////////////// Branch and Bound Path Finding Algorithm //////////////////// ----- #####

# Importing Libraries
import numpy as np
import pandas as pd
import operator
import matplotlib.pyplot as plt
import heapq
from sklearn.cluster import KMeans

# Setting Constants
crystal_active = np.ones((21,1), dtype=bool)
_MARINE = [30, 30]
position = 0
B = 0
bb_path = [[0]]
pq_list = []
min_b = 0
j = 0

# Defining Check-Function for path valuation
def reinitialise_crystal_active(path = [0], crystal_active = crystal_active):
    global position
    crystal_active[:] = True
    crystal_active[path] = False
    position = path[-1]
    return

# Function to find equivalent sub-optimal paths in bb_path
def equivalent_path_finder(path, bb_path):
    for comparison in bb_path:
        if comparison[-1] == path[-1]:
            if set(comparison) == set(path):
                return True
    return False

# Function to extend bb_path with a new path as well as a new value in bb_path
def extend_path(min_b, pq_list, j, stored_b):
    extension = [(dist_matrix.iloc[crystal_active[:, 0], position].idxmin(0))]

    # Updates pb min to new pb at the min_b index
    crystal_active[extension] = False
    new_b = stored_b - dist_matrix.iloc[extension, position].min(0) + dist_matrix.iloc[crystal_active[:, 0], position].min(0)
    if sum(crystal_active) != 1:
        heapq.heappush(pq_list, (new_b, min_b))
    if np.isnan(new_b):
        new_b = B
    
    # Extends bb_path to new node
    bb_path.append(bb_path[min_b] + extension)
    id_path_found = False
    id_path_found = equivalent_path_finder(bb_path[-1], bb_path[:-1])

    # Add new distance value to pb_list
    reinitialise_crystal_active(bb_path[-1])
    if id_path_found == False:
        heapq.heappush(pq_list, (stored_b + dist_matrix.iloc[crystal_active[:, 0], position].min(0), j + 1))
    else:
        heapq.heappush(pq_list, (B, j + 1))

# Function to black out existing extentions
def get_existing_extentions(path, bb_path = bb_path):
    extentions = [0]
    for other_path in bb_path:
        if other_path[:-1] == path:
            extentions.append(other_path[-1])
    return extentions

# K-Means Identification of Crystal Position
coords_array = np.loadtxt('coordinates1.txt')
cryst_centers = KMeans(20, 'k-means++', 15).fit(coords_array).cluster_centers_
cryst_centers = np.vstack((_MARINE, cryst_centers))

# Generating Euclidian Distance Matrix (EDM)
dist_matrix = pd.DataFrame(np.sqrt((np.array([cryst_centers[:,0]])-np.array([cryst_centers[:,0]]).T)**2 + (np.array([cryst_centers[:,1]])-np.array([cryst_centers[:,1]]).T)**2))

##### ----- Heuristic Path ----- ######
heuristic_path = [0]
reinitialise_crystal_active(heuristic_path)
for i in range(0, 20):
    B = B + dist_matrix.iloc[crystal_active[:, 0], position].min(0)
    position = dist_matrix.iloc[crystal_active[:, 0], position].idxmin(0)
    heuristic_path.append(position)
    reinitialise_crystal_active(heuristic_path)

##### ----- Branch and Bound Path ----- ######
reinitialise_crystal_active([0])
heapq.heappush(pq_list,(dist_matrix.iloc[crystal_active[:, 0], position].min(0),0))
for j in range(0, 10000):
    stored_b, min_b = heapq.heappop(pq_list)
    reinitialise_crystal_active(get_existing_extentions(bb_path[min_b]) + bb_path[min_b])
    if get_existing_extentions == True:
        heapq.heappush(pq_list, (B, min_b))
    extend_path(min_b, pq_list, j, stored_b)
    print(stored_b)