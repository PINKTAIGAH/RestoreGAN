lambda_jitter_array=($(seq 50 50 200))

for i in ${lambda_jitter_array[@]}; do 
    echo "# Adding $i as LAMBDA_JITTER" >> config.py
    echo "LAMBDA_JITTER = $i" >> config.py
    echo "EVALUATION_IMAGE_FILE= '../evaluation/jitter_$i'" >> config.py
    mkdir ../evaluation/jitter_$i
    # python3 train.py 
    # python3 evaluation.py
done
