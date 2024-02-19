# Environment Configuration on Rivanna

February 18, 2024 

## 1. Login

You need UVa VPN to access Rivanna. The standard one “UVa Anywhere” should work. Then, login to the server using the following command

```bash
ssh [computing-ID]@rivanna.hpc.virginia.edu

```

with the password for your computing ID. 

To avoid any unexpected interruption on the configuration process, we can use a `tmux` session

```bash
tmux new -s llmtuning 
```

In case of any interruption, we can log back in using the following command and continue the configuration

```bash
tmux a -t llmtuning
```

## 2. Load Modules

Before running anything, you need to load some relevant modules as you did in the CS department servers. For Rivanna, the module names are slightly different

```bash
module load anaconda cuda cudnn tmux
```

Using the following command to identify the version of cuda that was loaded via the previous command

```bash
module avail cuda
```

The result on my side is 

[https://www.notion.so](https://www.notion.so)

That means the loaded version is `cuda/12.2.2`.  

Another way to figure out the cuda version is to use the following command after load the cuda and cudnn modules

```bash
nvcc -V
```

## 3. Create a Virtual Python Environment

Assume the virtual environment name is `test` and the Python version is `3.9`, we use 

```bash
conda create -n test python=3.9
```

**Optional**: Once you create an environment, it may ask you to run `conda init`. It will add the following code block to your `.bashrc`, if you don’t want `conda` to activate any default environment, you can comment out the block

```bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/apps/software/standard/core/anaconda/2020.11-py3.8/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/apps/software/standard/core/anaconda/2020.11-py3.8/etc/profile.d/conda.sh" ]; then
        . "/apps/software/standard/core/anaconda/2020.11-py3.8/etc/profile.d/conda.sh"
    else
        export PATH="/apps/software/standard/core/anaconda/2020.11-py3.8/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```

## 4. Configuration for LLM Finetuning

Enter the `test` virtual environment

```bash
conda activate test
```

Install the Pytorch version that is pre-compiled with the cuda version

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12 -c pytorch -c nvidia
```

Note that, the value of `pytorch-cuda` should be aligned with the cuda version. It may take several minutes to finish the installation. 

### 4.1 Test the GPU Access

To test whether PyTorch can access GPUs on Rivanna, create a simple Python script `test.py` as the following 

```python
import torch

print(torch.cuda.is_available())
```

Submit it using the following example SLURM script `test.slurm`

```bash
#!/bin/bash -l                         

# --- Resource related ---             
#SBATCH --ntasks=1                     
#SBATCH -A [user-group-name]                      
#SBATCH -t 10:00 # Minute:Second
#SBATCH -p gpu # Partation type
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --mem-per-cpu=4GB # CPU memory

# Load the modules
module load anaconda cuda cudnn

# Activate the virtual Python environment
conda activate test

# Run the test code
python test.py
```

and the following command

```bash
sbatch test.slurm
```

If everything works well, the printed result from the simple Python script should be `True`. 

### 4.2 Install Additional Packages

Install the additional packages for the llm-finetuning package

```bash
pip install -r rivanna_requirements.yml
```

## 5. Run a Demo Example

The `[demo.py](http://demo.py)` file contains a simple example of fine-tuning Llama2 for sentiment classification. On Rivanna, you can submit the job using the following SLURM script, which is not very different from the previous one. 

```bash
#!/bin/bash -l                                                                                                                                              

# --- Resource related ---                                                                                                                                  
#SBATCH --ntasks=1                                                                                                                                          
#SBATCH -A uvailp                                                                                                                                           
#SBATCH -t 1:00:00 # Hour:Minute:Second                                                                                                                       
#SBATCH -p gpu # Partation type                                                                                                                             
#SBATCH --gres=gpu:1                                                                                                                                        
#SBATCH --mem-per-cpu=4GB # CPU memory                                                                                                                      

# Load the modules                                                                                                                                          
module load anaconda cuda cudnn

# Activate the virtual Python environment                                                                                                                   
conda activate test

# Run the test code                                                                                                                                         
python demo.py

# Or, run the training part using 
# python demo.py --task train
```
