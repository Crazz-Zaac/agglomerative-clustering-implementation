import streamlit as st
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from settings import DATASET_DIR as dataset
from settings import IMAGE_DIR 
from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets import make_blobs 


def plotFinalCluster(colors, cluster_labels, agg_cars):
	fig2 = plt.figure(figsize=(16,10))
	for color, label in zip(colors, cluster_labels):
	    subset = agg_cars.loc[(label,),]
	    for i in subset.index:
	        plt.text(subset.loc[i][0]+5, subset.loc[i][2], 'type='+str(int(i)) + ', price='+str(int(subset.loc[i][3]))+'k')
	    plt.scatter(subset.horsepow, subset.mpg, s=subset.price*20, c=color, label='cluster'+str(label))
	plt.legend()
	plt.title('Clusters')
	plt.xlabel('horsepow')
	plt.ylabel('mpg')
	plt.savefig(IMAGE_DIR+'/'+'plotFinalCluster.jpg')
	st.pyplot(fig2)

def plotCluster(pdf, agglom, agg_cars):
	import matplotlib.cm as cm
	n_clusters = max(agglom.labels_)+1
	colors = cm.rainbow(np.linspace(0, 1, n_clusters))
	cluster_labels = list(range(0, n_clusters))

	# Create a figure of size 6 inches by 4 inches.
	fig1 = plt.figure(figsize=(16,14))

	for color, label in zip(colors, cluster_labels):
	    subset = pdf[pdf.cluster_ == label]
	    for i in subset.index:
	            plt.text(subset.horsepow[i], subset.mpg[i],str(subset['model'][i]), rotation=25) 
	    plt.scatter(subset.horsepow, subset.mpg, s= subset.price*10, c=color, label='cluster'+str(label),alpha=0.5)
	#    plt.scatter(subset.horsepow, subset.mpg)
	plt.legend()
	plt.title('Clusters')
	plt.xlabel('horsepow')
	plt.ylabel('mpg')
	plt.savefig(IMAGE_DIR+'/'+'plotCluster.jpg')
	st.pyplot(fig1)

	plotFinalCluster(colors, cluster_labels, agg_cars)





def main():
	cell_df = pd.read_csv('dataset/cars_clus.csv')
	pdf = pd.read_csv('dataset/cars_clus.csv')
	num = 6
	st.title("Agglomerative clustering Implementation")
	st.write("")

	# st.image('img/svm1.jpg')

	st.sidebar.title("Evaluating different parameters")
	st.sidebar.subheader("View dataset")
	num = st.sidebar.number_input("Choose number of data to view", 5, 30)


	pdf[[ 'sales', 'resale', 'type', 'price', 'engine_s',
	       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
	       'mpg', 'lnsales']] = cell_df[['sales', 'resale', 'type', 'price', 'engine_s',
	       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
	       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
	pdf = pdf.dropna()
	pdf = pdf.reset_index(drop=True)

	if st.sidebar.checkbox('Show data'):
		st.write(pdf.head(num))
		st.write("Shape of data(before cleaning): ", cell_df.shape)

	#cleaning dataset by dropping 'null' value
		st.write("Shape of data(after cleaning): ", pdf.shape)


	#selecting featureset
	featureset = pdf[['engine_s',  'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]


	#Normalization (translates each feature individually such that it is between zero and one)
	from sklearn.preprocessing import MinMaxScaler
	x = featureset.values #returns a numpy array
	min_max_scaler = MinMaxScaler()
	feature_mtx = min_max_scaler.fit_transform(x)


	#Choosing a library
	lib = st.sidebar.selectbox('Library selection',('Choose a library ', 'SciKit Learn', 'SciPy'))
	if lib == 'SciPy':
		#importing necessary libraries
		import pylab
		import scipy.cluster.hierarchy
		from scipy.cluster.hierarchy import fcluster


		# def llf(id):
		#     return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])) )
		
		leng = feature_mtx.shape[0]
		D = np.zeros([leng,leng])
		for i in range(leng):
			for j in range(leng):
				D[i,j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j])
		
		method = st.sidebar.selectbox('Methods', ('Select a method', 'Single', 'Complete', 'Weighted', 'Average', 'Centroid'))
		if method == 'Average':
			Z = hierarchy.linkage(D, 'average')
		
			if st.sidebar.button("View Dendrogram"):
				fig = pylab.figure(figsize=(10,20))
				def llf(id):
				    return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])) )
				    
				dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =9, orientation = 'right')
				plt.savefig(IMAGE_DIR+'/'+'dendrogram.jpg')
				st.pyplot(fig)

			#selecting criterion
			crt = st.sidebar.selectbox('Select criterion', ('Choose a criterion', 'Distance', 'Maxclust'))
			#selecting number of clusters
			num = st.sidebar.number_input('Choose number of clusters', 5, 10)
			if crt == 'Distance':
				clusters = fcluster(Z, num, criterion='distance')
			elif crt == 'Maxclust':
				clusters = fcluster(Z, num, criterion='maxclust')
			else:
				st.write("Please choose a criterion")	

		elif method == 'Single':
			Z = hierarchy.linkage(D, 'single')
		
			if st.sidebar.button("View Dendrogram"):
				fig = pylab.figure(figsize=(10,20))
				def llf(id):
				    return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])) )
				    
				dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =9, orientation = 'right')
				plt.savefig(IMAGE_DIR+'/'+'dendrogram.jpg')
				st.pyplot(fig)

			#selecting criterion
			crt = st.sidebar.selectbox('Select criterion', ('Choose a criterion', 'Distance', 'Maxclust'))
			#selecting number of clusters
			num = st.sidebar.number_input('Choose number of clusters', 5, 10)
			if crt == 'Distance':
				clusters = fcluster(Z, num, criterion='distance')
			elif crt == 'Maxclust':
				clusters = fcluster(Z, num, criterion='maxclust')
			else:
				st.write("Please choose a criterion")


		elif method == 'Complete':
			Z = hierarchy.linkage(D, 'complete')
		
			if st.sidebar.button("View Dendrogram"):
				fig = pylab.figure(figsize=(10,20))
				def llf(id):
				    return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])) )
				    
				dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =9, orientation = 'right')
				plt.savefig(IMAGE_DIR+'/'+'dendrogram.jpg')
				st.pyplot(fig)
				
			#selecting criterion
			crt = st.sidebar.selectbox('Select criterion', ('Choose a criterion', 'Distance', 'Maxclust'))
			#selecting number of clusters
			num = st.sidebar.number_input('Choose number of clusters', 5, 10)
			if crt == 'Distance':
				clusters = fcluster(Z, num, criterion='distance')
			elif crt == 'Maxclust':
				clusters = fcluster(Z, num, criterion='maxclust')
			else:
				st.write("Please choose a criterion")	


		elif method == 'Weighted':
			Z = hierarchy.linkage(D, 'weighted')
		
			if st.sidebar.button("View Dendrogram"):
				fig = pylab.figure(figsize=(10,20))
				def llf(id):
				    return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])) )
				    
				dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =9, orientation = 'right')
				plt.savefig(IMAGE_DIR+'/'+'dendrogram.jpg')
				st.pyplot(fig)
				
			#selecting criterion
			crt = st.sidebar.selectbox('Select criterion', ('Choose a criterion', 'Distance', 'Maxclust'))
			#selecting number of clusters
			num = st.sidebar.number_input('Choose number of clusters', 5, 10)
			if crt == 'Distance':
				clusters = fcluster(Z, num, criterion='distance')
			elif crt == 'Maxclust':
				clusters = fcluster(Z, num, criterion='maxclust')
			else:
				st.write("Please choose a criterion")

		elif method == 'Centroid':
			Z = hierarchy.linkage(D, 'centroid')
		
			if st.sidebar.button("View Dendrogram"):
				fig = pylab.figure(figsize=(10,20))
				def llf(id):
				    return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])) )
				    
				dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =9, orientation = 'right')
				plt.savefig(IMAGE_DIR+'/'+'dendrogram.jpg')
				st.pyplot(fig)
				
			#selecting criterion
			crt = st.sidebar.selectbox('Select criterion', ('Choose a criterion', 'Distance', 'Maxclust'))
			#selecting number of clusters
			num = st.sidebar.number_input('Choose number of clusters', 5, 10)
			if crt == 'Distance':
				clusters = fcluster(Z, num, criterion='distance')
			elif crt == 'Maxclust':
				clusters = fcluster(Z, num, criterion='maxclust')
			else:
				st.write("Please choose a criterion")

	elif lib == 'SciKit Learn':
		dist_matrix = distance_matrix(feature_mtx,feature_mtx)
		lnk = st.sidebar.selectbox('Select linkage', ('Choose linkage', 'Ward', 'Complete', 'Average', 'Single'))
		if lnk == 'Ward':
			k = st.sidebar.number_input("Choose number of clusters", 5, 10)
			agglom = AgglomerativeClustering(n_clusters = k, linkage = 'ward')
			agglom.fit(feature_mtx)
			pdf['cluster_'] = agglom.labels_
			st.write("Number of cases in each group")
			st.write(pdf.groupby(['cluster_','type'])['cluster_'].count())
			st.write("Characterstics of each cluster")
			agg_cars = pdf.groupby(['cluster_','type'])['horsepow','engine_s','mpg','price'].mean()
			st.table(agg_cars)

			plotCluster(pdf, agglom, agg_cars) #for plotting 

		elif lnk == 'Complete':
			k = st.sidebar.number_input("Choose number of clusters", 5, 10)
			agglom = AgglomerativeClustering(n_clusters = k, linkage = 'complete')
			agglom.fit(feature_mtx)
			pdf['cluster_'] = agglom.labels_
			st.write("Number of cases in each group")
			st.write(pdf.groupby(['cluster_','type'])['cluster_'].count())
			st.write("Characterstics of each cluster")
			agg_cars = pdf.groupby(['cluster_','type'])['horsepow','engine_s','mpg','price'].mean()
			st.table(agg_cars)

			plotCluster(pdf, agglom, agg_cars) #for plotting 

		elif lnk == 'Average':
			k = st.sidebar.number_input("Choose number of clusters", 5, 10)
			agglom = AgglomerativeClustering(n_clusters = k, linkage = 'average')
			agglom.fit(feature_mtx)
			pdf['cluster_'] = agglom.labels_
			st.write("Number of cases in each group")
			st.write(pdf.groupby(['cluster_','type'])['cluster_'].count())
			st.write("Characterstics of each cluster")
			agg_cars = pdf.groupby(['cluster_','type'])['horsepow','engine_s','mpg','price'].mean()
			st.table(agg_cars)

			plotCluster(pdf, agglom, agg_cars) #for plotting 

		elif lnk == 'Single':
			k = st.sidebar.number_input("Choose number of clusters", 5, 10)
			agglom = AgglomerativeClustering(n_clusters = k, linkage = 'single')
			agglom.fit(feature_mtx)
			pdf['cluster_'] = agglom.labels_
			st.write("Number of cases in each group")
			st.write(pdf.groupby(['cluster_','type'])['cluster_'].count())
			st.write("Characterstics of each cluster")
			agg_cars = pdf.groupby(['cluster_','type'])['horsepow','engine_s','mpg','price'].mean()
			st.table(agg_cars)

			plotCluster(pdf, agglom, agg_cars) #for plotting 

	else:
		st.write("Please select one of the libraries")
		


if __name__=='__main__':
	main()


