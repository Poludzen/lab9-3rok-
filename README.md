# lab9-3rok-
## In this paper we want to compare: is GPU(CUDA) really faster than CPU on the same operations on vectors(code was given)
### First: installation CUDA on GoogleColab 
To install cuda we need to run this codes:
```
!apt-get --purge remove cuda nvidia* libnvidia-*
!dpkg -l | grep cuda- | awk '{print $2}' | xargs -n1 dpkg --purge
!apt-get remove cuda-*
!apt autoremove
!apt-get update
```
Then:
```
!wget https://developer.nvidia.com/compute/cuda/9.2/Prod/local_installers/cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64 -O cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64.deb
!dpkg -i cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64.deb
!apt-key add /var/cuda-repo-9-2-local/7fa2af80.pub
!apt-get update
!apt-get install cuda-9.2
```
Next check installed version:
```
!nvcc --version
```
load plugin
```
%load_ext nvcc_plugin
```
Now we are done! To run code with CUDA we need to paste  ```%%cu``` in the start of the cell 

### What length of vectors was in test?
This lengths were choosen : 10,50,100,200,500,1000,5000,10000,50000,100000,200000,500000,1000000

### Result of calculations:
##### Table of results
![alt text](https://github.com/Poludzen/lab9-3rok-/blob/main/images/cuda_vs_cpu_table.jpg?raw=True "Results")
#### Diagrams of results:
##### On counts : 10,50,100,200
![alt text](https://github.com/Poludzen/lab9-3rok-/blob/main/images/cuda_vs_cpu_time-1.jpg?raw=True "Diagram 1")
##### On counts : 10,50,100,200,500,1000
![alt text](https://github.com/Poludzen/lab9-3rok-/blob/main/images/cuda_vs_cpu_time-1.jpg?raw=True "Diagram 2")
##### On all counts:
![alt text](https://github.com/Poludzen/lab9-3rok-/blob/main/images/cuda_vs_cpu_time-1.jpg?raw=True "Diagram 3")

