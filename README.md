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


get the first version of code

Now, the 3D rendering of firing, is like waves in physics, that one point of light is just a week stirring to get the waves on, but the inner waves seems is the dominate power; so what I want now is let your network focus more on the input, now inner waves. 

after observing 4000 steps, I feel its just random in the inner waves of firing, the shaped like water waves

Get the second version of code


python3 02_01_cube_net.py


20250614 0144 PT
waves are still there, I heard you increase front to back, and decrease back to front, that's good, I feel, the fire should not be controlly globally, like biology neurons, it controled inside, one neuron should not fire too many times in one time peorid, not that controled how many percent globally? we do this experiment to mimic brain





python3 02_02_cube_net.py --bias 0.04 --d_theta 0.2 --tau_ref 2



python3 02_02_cube_net.py \
    --bias 0.008 \
    --theta0 1.0 \
    --d_theta 0.35 \
    --tau_ref 3 \
    --rho 0.8


20250614 1841 PT
what I am thinking now is that neural fire speed should be 8 times bigger than the input change, I mean when input moves 1, neural activity should be updated 8 times, which correspond to human time sense. In this way, we can also have the input connection afterwards reinforced better since we do not have enough firing now, because the input moves too fast for the neural to learn. give me complete code and ways to run it.

python3 02_03_cube_net.py


fire rate too high now, give me a version of code lower than this, but higher than before which I say too low

python3 02_04_cube_net.py


then fire rates get too low, Can you think of way to balance fire rate and let it focus on input not inner waves like firing?

python3 02_05_cube_net.py


20250614 2051 PT
one thing, for this 48*48*48 cube, since we want learn to repeat pattern in one layer, I feel, we should not keep isotropic or it will result in big waves back and forth, this is not good. Lets consider this, from the front to the back (output), we make the links sparse, keep the links between each layer as before, cancel the back to front links, but add a residual or recurrent layer to a specific layer to return the fire a little.

python3 03_01_cube_net.py