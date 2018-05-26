
import sqlite3
import matplotlib.pyplot as plt
#import matplotlib as mpl 
#mpl.use("gtk")
#import mpl.pyplot as plt
import numpy as np
#import matplotlib.rcsetup as rcsetup
#from numba.tests.npyufunc.test_ufunc import dtype
from pprint import pprint 
from prettytable import PrettyTable

def printo(object):
    pprint(vars(object))


class DataSet:
    
    def __init__(self,dbfile = None):
        self.dbfile = dbfile 
        
        # connect to database 
        self.connect_to_db()

    
    def connect_to_db(self,dbfile = None):
        if not( dbfile is None):
            self.dbfile = dbfile
        if not( self.dbfile is None):
            self.conn = sqlite3.connect(self.dbfile) 
            self.c = self.conn.cursor() 
        
            
    def load_data(self):
        
        self.ntrain = 200
    
        # CREATE NUMPY ARRAYS FOR DATA 
    
        # list of substations
        self.slist = ['CLAR','AMHE','WINL','BOWM','TROY','MAPL','GRAN','WAUT','CROSS']
        # list of variable names
        self.vlist = []
        for s in self.slist:
            self.vlist.append(s + '_VOLT')
            self.vlist.append(s + '_ANG')
        
        
        self.train_raw = create_array(self.c,self.vlist,'measurements',200)
        tmp = scale_data(self.train_raw)

        self.train = tmp[0]
        self.minvals = tmp[1]
        self.maxvals = tmp[2]
        
        # scale test data using same limits as training data
        self.test_raw = create_array(self.c,self.vlist,'analog_values',20)
        
        tmp = scale_data(self.test_raw, self.minvals,self.maxvals)
        self.test = tmp[0]
        
        #self.train = scale_data(create_array(self.c,self.vlist,'measurements',200))
        #self.test = scale_data(create_array(self.c,self.vlist,'analog_values',20))

# class for running a clustering algorithm
class ClusteringRun:
    
    def __init__(self,dataset = None):
        
        self.db = dataset 
        self.J = 0
        self.labels = []
        self.label_idx = []
        self.vlist = []
        self.sorted_labels = []
        
    
    def do_clustering(self,k = 4,cluster_vars = None):
        debug = False
        

        # cluster along given dimension
        if (type(cluster_vars) is list) and cluster_vars.__len__() > 0: 
            vlist = cluster_vars 
        else:
            # cluster along all non-singelton dimensions 
            range_tol = 1e-5
            vlist = []
            for v in self.db.train.dtype.names:
                if self.db.maxvals[v] - self.db.minvals[v] > range_tol:
                    vlist.append(v)
            if debug:
                print(vlist)
        self.vlist = vlist 
            
        show_plots = False
        plot_scaled_data = False
        tol = 1e-5    
            
        # randomly assign each data point to cluster
        cluster = np.random.randint(low=0,high=k,size=self.db.ntrain)
        
        # compute center of clusters
        u = compute_centroids(self.db.train, cluster, k)
        
        # compute error function
        J = compute_cluster_deviation(self.db.train, cluster, u, vlist)
        if debug:
            print(J)
        diff = 1
        
        while diff > tol:
            #  assign data to centroids and compute error
            res = assign_centroids(self.db.train, u, vlist)
            
            diff = J - res[1]
            J = res[1]
            if debug:
                print(J)
                        
            u = compute_centroids(self.db.train, res[0], k)
            
        # store cluster assignment and centroids     
        self.clusters = res[0]
        self.u = u
        self.J = J
        # centroids for unscaled data
        self.u_raw = unscale_centroids(self.u, self.db.minvals, self.db.maxvals)
        
    # Classify the test data into clusters    
    def classify_test_data(self,show_output = True):
        
        res = assign_centroids(self.db.test,self.u,self.vlist)
        self.test_clusters = res[0]
    
        if show_output: # print table of data points and assigned clusters
            t = PrettyTable(["row","cluster",'cluster nr','AMHE_ANG'])
            for i in range( np.size( self.test_clusters )):
                t.add_row([i,self.sorted_labels[self.test_clusters[i]],self.test_clusters[i],
                           self.db.test_raw['AMHE_ANG'][i]])
            print(t)

            
    def plot_results(self,plot_scaled_data=False,lines=[],plist = None):
        debug = False
        # plot clusters over pairs of variables
        if plist is None:
            plist = [] 
            for node in self.db.slist:
                plist.append(['{0}_ANG'.format(node),'{0}_VOLT'.format(node)])
            
        if not( self.sorted_labels == []):
            labels = self.sorted_labels
            if debug:
                print("Plot results:") 
                print(self.sorted_labels)
                print(self.label_idx)
                print(self.labels)
                print(labels)
                for c in self.u_raw:
                    print("{0}".format(c['AMHE_ANG']))
        else:   
            labels = ['{0}'.format(x) for x in range(self.u.__len__())]

        # final cluster plots
        for p in plist:
            plt.figure()
            if plot_scaled_data:
                plot_clusters(self.db.train,self.u,self.clusters,p,labels = labels)
            else:
                plot_clusters(self.db.train_raw,self.u_raw,self.clusters,p, labels = labels)
                plt.plot(self.db.test_raw[p[0]],self.db.test_raw[p[1]],'k*')
                
        
        for l in lines:
            plt.figure()
            plot_voltage_diff(self.db.train_raw, self.u_raw, self.clusters,l, labels = labels)
            
    
    def label_clusters(self):
      
        debug = False
        
        centroids = [c for c in self.u_raw]
        self.labels = ['Peak load','Low load','WINL generator shut down','MAPL-GRAN line disconnected']
        label_idx = []

        # peak load is when CLARK-BOWMAN angle difference is at the maximum
        peak_idx = np.argmax([c['CLAR_ANG']-c['BOWM_ANG'] for c in centroids] )
        c = centroids.pop(peak_idx)

        label_idx.append(peak_idx)
        
        # low load is when CLARK-BOWMAN angle difference is at the minimum 
        idx = np.argmin([c['CLAR_ANG']-c['BOWM_ANG'] for c in centroids] )
        c = centroids.pop(idx) 

        for i in range(self.u_raw.__len__()):
            if self.u_raw[i] == c:
                label_idx.append(i)
                 
                
        # generator shut down is when WINLOCK-MAPLE angle difference is 0 
        idx = np.argmin( [c['WINL_ANG']-c['MAPL_ANG'] for c in centroids] )
        c = centroids.pop(idx) 

        for i in range(self.u_raw.__len__()):
            if self.u_raw[i] == c:
                label_idx.append(i) 
        
        # only remaining cluster is line outage 
        for i in range(self.u_raw.__len__()):
            if self.u_raw[i] == centroids[0]:
                label_idx.append(i)
        
        #print(label_idx)
        self.label_idx = label_idx 
        
        sorted_labels = []
        k = self.u.__len__()
        for i in range(k):
            for idx in range(k):
                if self.label_idx[idx] == i:
                    sorted_labels.append(self.labels[idx])
        self.sorted_labels = sorted_labels 
        if debug:
            print(self.labels)
            print(self.label_idx)
            print(self.sorted_labels)
# scale all variables in structured array
def scale_data(data,vmin = None, vmax = None, tol = 1e-10):
    scaled_data = np.zeros(np.size(data),dtype = data.dtype)
    
    if vmin is None:
        vmin = {}
        vmax = {}
        compute_minmax = True
    else:
        compute_minmax = False
        
    for v in data.dtype.names: 
        if compute_minmax:
            vmin[v] = np.min(data[v])
            vmax[v] = np.max(data[v])
        
        vrange = vmax[v] - vmin[v] 
        if vrange > tol:
            scaled_data[v] = (data[v]-vmin[v])/vrange
        else:
            scaled_data[v] = data[v]-vmin[v] 
    
    return (scaled_data,vmin,vmax)
    

# read data from table in numpy array 
def create_array(c,varlist,table,n):
    
    data = np.zeros(n,dtype = [(v,'float32') for v in varlist] )
    #data = np.zeros(n,dtype = [(varlist[i],datalist[i]) for i in range(0,varlist.__len__())])
    
    #print(data)
    for i in range(0,n):
        # retrieve data for this time step
        c = c.execute('SELECT name,value FROM {1} WHERE time == {0}'.format(i+1,table))
        # store data in array
        for row in c:
            data[row[0]][i] = row[1]
    return data 

# Return list of centroids
def compute_centroids(data,cluster,k):
    debug = False
    
    centroids = []
    for i in range(0,k):
        if debug:
            print("Cluster {0}".format(i))
        c = {}
        # pick out elements in cluster
        cidx = np.where(cluster == i)
        for v in data.dtype.names:
            c[v] = np.mean(data[cidx][v])
            if debug:
                print("{0}: {1}".format(v,c[v]))
            #print(cdata)
        if debug:
            print(c)
        #print(np.mean(data[cidx],axis=0))
        #print(np.mean(data[cidx],axis=1))
        centroids.append(c)
    return centroids

# assign all data points to closest cluster centroid and compute cluster deviation
def assign_centroids(data,centroids,varlist):
    debug = False
    
    k = centroids.__len__() # number of centriods 
    n = np.size(data) # number of rows 
    
    dist = np.zeros([n,k])
    for i in range(0,k): 
        # compute euclidean distance to centriod i
        for v in varlist:
            # add square for this variable
            dist[:,i] = dist[:,i] + np.square(data[v]-centroids[i][v])
        #dist[:,i] = np.linalg.norm()
        # square root
        dist[:,i] = np.sqrt( dist[:,i] )
        
    # select closes centriod
    cluster = np.argmin(dist,axis=1)
    
    # put random data point into empty cluster
    samples = list(range(0,n))
    for i in range(0,k):
        if not(i in cluster):
            rnds = np.random.choice(samples)
            cluster[ rnds ] = i 
            samples.remove(rnds)
    
    
    if debug:
        print(cluster)
        print(dist[list(range(n)),cluster])
        print(np.size(dist[list(range(n)),cluster]))
        
    return (cluster,np.sum(dist[list(range(n)),cluster]))

# given centroids and cluster assignment, compute cluster deviation
def compute_cluster_deviation(data,cluster,centroids,varlist):
    
    debug = False
    
    k = centroids.__len__() # number of centriods 
    n = np.size(data) # number of rows 
    
    dist = np.zeros(n)
    # add squared contribution from each variable
    for i in range(0,k):
        
        for v in varlist: 
            
            dist[cluster == i] = dist[cluster == i] + np.square( 
                                    data[v][cluster == i] - centroids[i][v] )
    if debug: 
        print(dist)
    return np.sum(np.sqrt(dist)) 
        

# plot graph of data and cluster centroids over dimesions in varlist (2 dimensions)
# data - structured array containing data points 
# centroids - list of dicitionaries with cluster centroids 
# cluster - array of assigned cluster for each data point 
# varlist - list of variable names for the dimensions to plot the clusters (at least 2 vars)
def plot_clusters(data,centroids,cluster = None,varlist = None,labels = None):
    
    k = centroids.__len__()
    if labels is None:
        labels = list(range(k))
    
    
    # assign data to clusters
    if cluster is None: 
        c = assign_centroids(data,centroids,varlist)
        cluster = c[0]
    
    # plot over first 2 dimensions
    plot_varlist = varlist[0:2]
    
    
    colors = ['C{0}'.format(i) for i in range(0,10)]
    # plot data of each cluster
    for i in range(0,k):
        ax = plt.plot(data[plot_varlist[0]][cluster == i],
                      data[plot_varlist[1]][cluster == i],
                      marker = '.',
                      color = colors[i],
                      linestyle = '',
                      label = '{0}'.format(labels[i]) )
    
    # plot cluster centres
    for i in range(0,k):
        plt.plot(centroids[i][plot_varlist[0]],centroids[i][plot_varlist[1]],
                 color = 'k',
                 marker = '*', 
                 markersize = 10)
        
    plt.grid()
    plt.legend()
    plt.xlabel(plot_varlist[0])
    plt.ylabel(plot_varlist[1])


# plot difference in voltage magnitudes and angles for two buses for different clusters 
def plot_voltage_diff(data,centroids,cluster,line,labels = None):
    
    volt_vars = []
    ang_vars = []
    for v in line:
        volt_vars.append(v+'_VOLT')
        ang_vars.append(v+'_ANG')
    
    
    k = centroids.__len__()
    if labels is None:
        labels = list(range(k))
        
    x = data[ang_vars[0]]-data[ang_vars[1]]
    y = data[volt_vars[0]]-data[volt_vars[1]]
    

    colors = ['C{0}'.format(i) for i in range(0,10)]
        
    for i in range(0,k):
        plt.plot( x[cluster == i], y[cluster == i],
                  marker = '.',
                  color = colors[i],
                  linestyle = '',
                  label = '{0}'.format(labels[i]) )
    plt.xlabel(ang_vars[0]+' - '+ang_vars[1])
    plt.ylabel(volt_vars[0]+ ' - '+volt_vars[1])
    plt.grid()
    plt.legend()


def unscale_centroids(centroids,minvals,maxvals, tol = 1e-10):
    unscaled_centroids = []
    for c in centroids:
        d = {}
        for v in minvals:
            vrange = maxvals[v] - minvals[v] 
            if vrange > tol:
                d[v] = c[v]*vrange + minvals[v] 
            else:
                d[v] = c[v] + minvals[v]
        unscaled_centroids.append(d)

    return unscaled_centroids
    
    
def main():
    plt.close()
    
    # plot angle differences for all lines
    lines = [['CLAR','BOWM'],
            ['AMHE','WAUT'],
            ['WINL','MAPL'],
           # ['BOWM','TROY'],
           # ['BOWM','CROSS'],
            ['TROY','MAPL'],
            ['MAPL','GRAN'],
            #['CROSS','WAUT'],
            #['WAUT','GRAN'],
            #['BOWM','MAPL']
            ]
    
    plist = [['WINL_ANG','WINL_VOLT'],
             ['AMHE_ANG','AMHE_VOLT'],
             ['CROSS_ANG','CROSS_VOLT'],
             ['MAPL_ANG','MAPL_VOLT']]
    
    #cluster_vars = ['MAPL_ANG','MAPL_VOLT']
    cluster_vars = []
    
    k = 4
    dbfile = 'data/subtables.sqlite'
    
    # load data from sqlite database
    db = DataSet(dbfile)
    db.load_data()
    
    # run several clusterings
    cluster_runs = []
    for i in range(10):
        c = ClusteringRun(db)
        c.do_clustering(k,cluster_vars)
        #print(c.J)
        cluster_runs.append(c)
        
    # find cluster with minimum error
    idx = np.argmin([c.J for c in cluster_runs])
    cl = cluster_runs[idx]
    cl.label_clusters()
    
    cl.classify_test_data()
    
    cl.plot_results(plot_scaled_data=False,lines=lines,plist = plist) 
    
    
    plt.show()
    

if __name__ == "__main__":
    main()
    
    
