# FusionDeformer 

## Installation

    conda create -y -n FusionDeformer python=3.9
    conda activate FusionDeformer
    pip3 install torch torchvision # install torch with cuda support
    conda install -y -c conda-forge igl
    pip install -r requirements.txt

## Usage
**NOTE:** This repository **requires** a GPU to run.

### Run examples
``main.py`` is the primary script to use. You may pass arguments using the ``--config`` flag, which takes the path to a ``.yml`` file. See ``example_config.yml`` for an example. Alternatively, you may pass command line arguments manually, which override the arguments provided by the config file. Below, we provide example usage:
    
    # Use all arguments provided by the example config
    python main.py --config configs/sds.yml

### Outputs
The outputs will be saved to the path specified in the run configuration, which is ``./outputs`` by default. The output folder will contain several folders: ``images`` contains intermittently saved samples of the rendered images passed to CLIP, ``logs`` will contain tensorboard logs of the optimization process, ``mesh_best_clip``, ``mesh_best_total``, and ``mesh_final`` contain the optimized meshes at the best CLIP score, the best total loss, and the final epoch. The configuration file is also saved at ``config.yml`` and a video of the optimization process is saved at ``video_log.mp4``. 

### Common bugs
#### Jacobian temp files
The ``NeuralJacobianFields`` code in this repository will create several temporary files in ``outputs/tmp``. Note that if these temporary files already exist, this code will attempt to read the existing files instead of overwriting them. This may cause issues if you run multiple examples with the same output path, intending to overwrite the output folder.
