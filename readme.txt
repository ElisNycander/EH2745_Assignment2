--------------------------------------------------------------------
--------------------------------------------------------------------
Overview of Python code for k-means clustering - Assignment 2 EH2745
--------------------------------------------------------------------
--------------------------------------------------------------------
by Elis Nycander 


-------------------------------------------------------------
CLASSES
-------------------------------------------------------------


----- DataSet ------

Connects to SQLITE database and reads data into numpy arrays 
used by the clustering algorithm. Normalizes data. 

-- Methods: 

* connect_to_db: setup connection to sqlite database

* load_data: read tables from database, create numpy arrays with scaled and unscaled data
  - calls scale_data() and create_array()

  
----- ClusteringRun --------

Run a clustering algorithm on training data, classify test data, and plot results. 
Uses the data from its DataSet object. 

-- Methods: 

* do_clustering: main clustering algorithm, randomly assign data to clusters 
  and iterate until there is no change in cluster assignment between iterations 
  - calls compute_centriods(), assign_centroids(), compute_cluster_deviation(), unscale_centroids()

* classify_test_data: classify the test data by assignment to the nearest cluster 
  - calls assign_centroids() 
  
* plot_results: plot graphs of clusters over given variables 
  - calls plot_clusters(), plot_voltage_diff() 
  
* label_clusters: use heuristics to associate the clusters with different scenarios


-------------------------------------------------------------
FUNCTIONS
-------------------------------------------------------------

-------- main -----------

Creates DataSet object, then uses this to run multiple clustering runs by 
ClusteringRun. The clustering with lowest error measure is selected, the test 
data classified and the results plotted. 


--------------------------------------------------------------
NOTES
--------------------------------------------------------------

1. The data from the database is saved as a structured numpy array. 
This means that the data can be accessed using the names of the variables, 
such as 

data['CROSS_ANG'][1:40] 

However, this also means that we must loop
over the different variables in several cases, such as when computing the 
distance of the data points to the centroids. It might be more efficient
if the data was instead saved as an ordinary numpy array. The column indices 
could then be stored in a dictionary, such as 

d={'CROSS_ANG':1,'CROSS_VOLT':2,...}

This migth be more efficient since looping over columns could in some places 
be avoided and instead done implicitly by numpy functions. 

2. The clusters are stored in a list of dictionaries 

clusters = [{'CROSS_ANG':24.455,'CROSS_VOLT':0.978,...},...]

Another option would be to store the centroids in a numpy array similar 
to how the data is stored. This might be more efficient in some places, but
the effect is probably not so large since the number of clusters is low 
compared to the dimension of the data. 

3. The clustering is done on normalized data, but the results can be displayed
either in terms of the normalized or the original data.   

4. The clustering algorithm itself is independent of the data used for the 
clustering. The only condition that must be fulfilled by do_clustering()
is that if it has been given a list of variables over which to cluster, cluster_vars,
then these variables must be present in the data it has loaded. However, when 
plotting the results it is assumed that the data  corresponds to the given system, 
so that e.g. it makes sense to plot the difference of voltage angles and similarly
when labeling the clusters according to the different load scenarios.  

-------------------------------------------------------------
VIDEO
-------------------------------------------------------------
Link to video:
https://kth.box.com/s/vemjtxsxywlwwlh0mu5ox3czcrq2taiv



