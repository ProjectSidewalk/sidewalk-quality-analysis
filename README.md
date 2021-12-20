# sidewalk-quality-analysis
In this project, we aim to infer the quality of Project Sidewalk users based on their interactions with our system--both low-level interactions like mouse clicks and moves--as well as higher-level, more application related interactions (amount of panning on a street view image, etc.). More details are in our Dropbox folder [`ProjectSidewalk_PredictingUserQuality`](https://www.dropbox.com/home/ProjectSidewalk_PredictingUserQuality)

## Running the notebook locally using Anaconda 

Follow the instructions below and consult the [Managing Environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) section in the conda docs for more details. There is also a nice conda cheetsheet [here](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf).

### Step 1: Open your Anaconda terminal and go to the src dir
On **Mac**, this should be as simple as opening `terminal` (or, for example, use [`iterm2`](https://iterm2.com/)—my preferred terminal program).

On **Windows**, open the `Anaconda Powershell Prompt`.

Make sure you are in the root directory of this project. For example, for me (on my Windows), this is:

```
> pwd
D:\git\sidewalk-quality-analysis
```

### Step 2: Create an environment from the environment.yml file

```
> conda env create -f environment.yml
```

This might take a few mins but should end with something like

```
done
#
# To activate this environment, use
#
#     $ conda activate sidewalk-quality-analysis
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

Optionally, if you'd like to list the active conda environments on your system and verify that the `a11y-qual-analysis` environment was created:

```
> conda env list
```

### Step 3: Activate the environment

```
> conda activate sidewalk-quality-analysis
```

### Step 4: Open jupyter notebook
Now you should see the command line prompt prefixed by the current environment: `(sidewalk-quality-analysis)`. So, your command prompt should look like the following or something similar:

```
(sidewalk-quality-analysis)$
```

Now you can type in `jupyter notebook` and find `analysis.ipynb`. 

```
(a11y-qual-analysis)$ jupyter notebook
```

In Jupyter Notebook environment, navigate to the `analysis.ipynb.ipynb` file and open it.

