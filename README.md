# EasyDML
An Easy programming tool for Distributed Machine Learning

```python
'
 ______                _____  __  __ _
|  ____|              |  __ \|  \/  | |
| |__   __ _ ___ _   _| |  | | \  / | |
|  __| / _` / __| | | | |  | | |\/| | |
| |___| (_| \__ \ |_| | |__| | |  | | |____
|______\__,_|___/\__, |_____/|_|  |_|______|
                  __/ |
                 |___/
'
```
## Introduction
This is a light python project for programming distributed machine learning algorithm easily, there are three key objects we want to achieve:
* Easy Programming. You can transform one machine learning algorithm distirbuted easily, which means that you will add as less codes as possible for the parallelization of one current sequential ML algorithm.
* Well Scalability. We support the increase of data and machines. You can process bigger data using more processors, which means that with the increase in the number of processors, the running time of your algorithm on the same data will decrease, and as long as your cluster is big enough, you can process as big data as you want in acceptable time
*

## Dependence
We use python to make this project can be deploy on windows, linux and mac, all dependences is listed as follow:  
-- python3  
-- pip  
-- numpy  
-- mpi4py  
-- python-gflags  
-- pytest  
You can use pip to install all the packages and easily deploy this project on your cluster or PC.

## How to Run
See the shell files in scripts


##path setting for different os
You need to change path in run.sh && alg/*.py correspond to different os.  
In windows, it's suggested to use spliter like "\",  
   e.g.  run.sh :  mpiexec -n 5 python alg\kmeans.py  
         kmeans.py : gflags.DEFINE_string('dataset', '.\data\iris_norm.csv', 'the input dataset')  
In Linux, it's suggested to use spliter like "/",  
   e.g.  run.sh : mpiexec -n 5 python3 ../alg/kmeans.py  
         kmeans.py : gflags.DEFINE_string('dataset', '../data/iris_norm.csv', 'the input dataset')  
