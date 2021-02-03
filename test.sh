source ~/anaconda3/bin/activate;
conda activate HPC;
cd ~/GitHub/DelineatorSwitchAndCompose;

# Get a list of all possible models
list_all_models=()
input="list_files.txt"
while IFS= read -r line
do
  list_all_models+=("$line")
done < "$input"

for i in {1..109}
do
    model=${list_all_models[$i]};
    echo $i/109 $model;
    echo "";  
    python3 test_12lead.py --basedir ~/DADES/DADES/Delineator/ --model_name ${model} --hpc 0 --database ludb;
    python3 test_12lead.py --basedir ~/DADES/DADES/Delineator/ --model_name ${model} --hpc 0 --database zhejiang;
    python3 test_holter.py --basedir ~/DADES/DADES/Delineator/ --model_name ${model} --hpc 0;
done
