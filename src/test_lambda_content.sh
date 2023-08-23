lambda_content_array=($(seq 50 50 200))

for i in ${lambda_content_array[@]}; do 
    echo "# Adding $i as LAMBDA_CONTENT" >> config.py
    echo "LAMBDA_CONTENT = $i" >> config.py
    echo "EVALUATION_IMAGE_FILE= '../evaluation/content_$i'" >> config.py
    mkdir ../evaluation/content_$i
    python3 train.py 
    python3 evaluation.py
done
