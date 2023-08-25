critic_iter_array=($(seq 2 1 6))

for i in ${critic_iter_array[@]}; do 
    echo "# Adding $i as DISCRIMINATOR_ITERATIONS" >> config.py
    echo "DISCRIMINATOR_ITERATIONS = $i" >> config.py
    echo "EVALUATION_IMAGE_FILE = '../evaluation/critic_iter_$i'" >> config.py
    mkdir ../evaluation/critic_iter_$i
    python3 train.py 
    python3 evaluation.py
done
