# launch 48 jobs of inner.sbatch
for j in $(seq 100)
do
  sbatch --job-name=$j.run --output=$j.out ./singlejob_submit.sh
done
