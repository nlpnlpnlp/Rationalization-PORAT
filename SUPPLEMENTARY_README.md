# PORAT (GAME EXAMPLE TRACED)


## Environments
Ubuntu 22.04.4 LTS; NVIDIA RTX6000 Ada; CUDA 12.1; python 3.9.

We suggest you to create a virtual environment with: conda create -n PORAT python=3.9.0

Then activate the environment with: conda activate PORAT 

Install packages: pip install -r requirements.txt


## Data
- An Example from Beer-Aroma. 



## Results

To intuitively demonstrate the game process of our proposed method, we conduct experiments on the Beer-Aroma dataset, where the text sequences in the Aroma aspect represent the rationale we expect. 

We introduce the policy intervention mechanism starting from the 200th epoch and visually tracked the game process of a sample.

We find that PORAT can gradually guide the model to escape from the representation distribution before the 200th epoch and progressively learn the rationale in the Aroma aspect. This is consistent with our previous conclusion. 

Without the intervention policy, the model will continue to maintain the state before the 200th epoch.


