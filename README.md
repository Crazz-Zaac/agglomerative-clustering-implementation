# Agglomerative Clustering Implementation

![alt text](https://github.com/Crazz-Zaac/agglomerative-clustering-implementation/blob/master/img/dendrogram.jpg =250x250)

## Algorithm:
```
1. Create n clusters, one for each data point
2. Compute the Proximity Matrix which wil be a (nxn) matrix
3. Repeat:
	i. Merge the closest clusters
	ii. Update the proximity matrix
  Until: Only a single cluster remains
```

## Getting ready
First setup virtual environment, activate it install the essentials.


## Install the essentials first
```bash
pip install -r requirements.txt
```

## Running the application
```bash
streamlit run app.py
```
Note: Make sure you're inside the root folder where ``` app.py``` is there.


## Acknowledgement 
Coursera course: Machine Learning with Python by IBM

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

