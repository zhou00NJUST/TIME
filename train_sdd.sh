for i in 9
do
    #python train.py --test_set $i --num_epochs 100 --eta_min 1e-5  --batch_size 32 --learning_rate 1e-3  --randomRotate True  --phase train --train_phase gen_memory --load_model 0 --memory_filter 1
    python train.py --test_set $i --num_epochs 110 --eta_min 1e-5  --batch_size 32 --learning_rate 1e-3  --randomRotate True  --phase train --train_phase train_adaptor --load_model 0 --load_memory 100
done
