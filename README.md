# Tempestas
Learning to drive in adverse weather conditions with Reinforcement Learning and CARLA

Exploration of the use of Reinforcement Learning in Autonomous vehicles with an aim towards teaching a vehicle to drive in multiple weather conditions from scratch.

Using [CARLA](https://carla.org/) and [gym-carla](https://github.com/cjy1992/gym-carla) (with some alterations) for the env.

Using [This Paper](https://arxiv.org/abs/1902.03765) along with [this paper](https://arxiv.org/abs/1807.01001) as a basis, with an eye towards further comparing the base model with more advanced models.


## Requirements 

To run this github you will need the following:

> Install requirments pytorch-requirment.txt


> Download and install [gym-carla](https://github.com/rbuckley25/gym-carla) fork built for this project


> Download and Install Carla 0.9.12 which can be found [here](https://github.com/carla-simulator/carla/releases/tag/0.9.12/)


> Download and install [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

## Running Demo, Evaluation and Carla Models

To run any of the Carla Models you will first need to start your carla simulator. Instructions for how to do this can be found [here](https://carla.readthedocs.io/en/latest/start_quickstart/#running-carla)

Once your Carla simulator is running you can run any of the model notebooks by simply 



## Image Segment and Latent Transfer

Image Segment and Latent Transfer can both be run just by running each of the cells in order. 
Unfortunatly due to the size of the datasets generated from cyclegan It is not possible to upload the data sets to Github, The datasets can be generated using the CycleGAN implementaion above using the cyclegan-wet and cyclegan-cloudysunset datasets to train the CycleGAN and the ClearNoon Dataset to generate the new dataset.


## Running CycleGAN

To run one of the trained cycleGAN models on your own datasets you can replace the the checkpoints folder in your CycleGAN repository with checkpoints folder in this repository and run the following command:


    `python test.py --dataroot PATH_TO_IMAGES_TO_CONVERT --name cloudysunsetcyclegan --model test --no_dropout --preprocess none --results PATH_TO_RESULTS_FOLDER --num_test NUMBER_OF_IMAGES_IN_DATASET`

To train a new model using the data in cyclegan-cloudysunset or cyclegan-wet use the follwing command


    `python train.py --dataroot ~/Carla/Project_Code/Tempestas/Data/cyclegan-cloudysunset --name NAMEOFYOURMODEL --model cycle_gan --preprocess none --n_epochs 20 --n_epochs_decay 10`

For more information about running CycleGAN or Pix-to-Pix please refer to the original [Github](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)