Create the env:
```
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```


20250612 0208 PT

python3 01_01_spiking_local.py         # 默认 n=48

This version of code works very well to just let the output layer mimic the input pattern. Just repeat. 
By observation, 
from step 0 - step 500, only random output, very little. Only mid layer connected directly to input fired. 
from step 500 - step 1800, the mid layer and output layer get into completely chaos. 
the chaos gradually be cleared out and only the input pattern remains. 
after step 2000, the output repeat the input very well. 

And I let ChatGPT-o3-Pro write a readme file for this script which names: Readme_archived_01_01_spiking_local.md

actually, only one layer of neurons in this scripts, it takes input then output into the result(seems 3 layers)
this layer of neurons has w_in, w_out, and w_rec(the neurons in this layer is connected with each other)

Only for w_in, the script employs the STDP mechanism.
For w_out it just use supervised from the result target error. 



20250614 0003 PT
Now for the next code, I want to cancel the input after step 3000 to see the following steps. 

python3 01_02_spiking_local.py 

Since it only has one layer neurons, after I canceled the input at step 3000 the output becomes chaos, though it learns faster than the beginning when I start the input again after step 4000. But this one layer neurons cannot remember the input pattern in a very good way. 


This version of code is saved as 1.0.0.

Now, what I want to do is to add more layers, like 4 layers, and between these neurons, I want have a spatial structure:
I am thinking of 2 proposals
1. image a lot of these middle layer neurons in a 3D cube of length ** 3. each neurons has their 3D coordinate. Along the traditional NN's forward connection (from front to the back), they have backward connection as well (from the back to the front), also, they have connections with neurons in the same layer. so for this cube, these connections' direction looks the same seen from any surface of this cube. The most important thing, is that they can only connect with their 3D neighboors, so like 1 neuron can at most connect to 6 neurons.

2. For proposal 1, a neuron connected with 6 neurons in one direction(with another direction it will be 12), the connections are much less than brain neurons (1k). So, based on proposal 1, we can have smaller cube inside this big 3D cube, which we can call them Groups_of_neurons. inside each group they are connected with all neurons inside this group. For group with group, they following the  only connection with neighboors. 


since proposal 2 is  more complex, let start with proposal 1. For proposal 1, make sure the isotropic for this cube of neurons, and the computation was not just one direction, (but we can compute step by step just more directions, actually we can lay the 3D into 1D, butDifferentiate between different links in different locations) ; these neurons , like 48*48*48 follow the STDP rules with each other like your code above. 