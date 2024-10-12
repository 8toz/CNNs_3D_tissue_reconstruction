#!/bin/bash
#SBATCH --job-name="initial_run"
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --time=0-01:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=4 
#SBATCH --gres=gpu:1 
#SBATCH --mail-type=END
#SBATCH --mail-user=noah.hoogenboom@xs4all.nl
#SBATCH --output=testMatlab.out

# Store current directory to MYSTART
MYSTART=`pwd`


# Define the name of the Matlab script
matScript=main
echo ${matScript}


cd ${TMPDIR}

# Copy files to the tempdir

echo "Copy files from the home/folder directory to the temporal directory"
cp -prf ${MYSTART}/* ${TMPDIR}

module purge
#module load 2021
#module load MATLAB/2021a-upd3
#module load ANSYS/2021R2

module load 2022
module load MATLAB/2022a-upd4
module load ANSYS/2023R2


# Create a list with the hostnames of the nodes assigned to us
NODEFILE=`mktemp`
scontrol show hostname | awk -v TPN=$SLURM_TASKS_PER_NODE '{ printf("%s:%d\n", $1, TPN) }' > ${NODEFILE}

echo "Contents of the nodefile"
cat ${NODEFILE}
echo

# Run Matlab 
matlab -nodisplay -nojvm -singleCompThread -r "${matScript} ; quit" 

# Remove temporary nodefile
rm -f ${NODEFILE}

# Copy output to results at Research Drive.
#echo "Copy results from the temporal directory to the Research Drive (results only)"
#rclone sync ${TMPDIR}/Results/${MODELNAME}/${FOLDER1}  ${RESULTPATH}/${MODELNAME}/${FOLDER1}

# Lines to copy the files from the home/folder directory to the temporal direction
echo "Copy files from the the temporal directory to the home/folder directory"
mkdir -p ${MYSTART}/output
cp -prf * ${MYSTART}/output



