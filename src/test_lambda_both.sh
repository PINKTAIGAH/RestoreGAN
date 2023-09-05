lambda_content_array=($(seq 50 50 200))
lambda_jitter_array=($(seq 50 50 200))

for i in ${lambda_content_array[@]}; do 
    for j in ${lambda_jitter_array[@]}; do 
        echo "LAMBDA_CONTENT = $i" >> config.py
        echo "LAMBDA_JITTER = $i" >> config.py
        echo "EVALUATION_IMAGE_FILE= '../evaluation/content_$i.$j'" >> config.py
        mkdir ../evaluation/content_$i.$j
        python3 train.py 
        cp ../models/gen.pth.tar /home/brunicam/GPFS/petra3/scratch/brunicam/RestoreGAN-models/gen.restore_$i.$j.tar 
        cp ../models/disc.pth.tar /home/brunicam/GPFS/petra3/scratch/brunicam/RestoreGAN-models/disc.restore_$i.$j.tar 
    done
done
