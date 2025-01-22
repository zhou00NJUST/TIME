for i in 2 3 
do
    #python train.py --test_set $i --num_epochs 1000 --eta_min 1e-5  --batch_size 32 --learning_rate 1e-3  --randomRotate True  --phase train --train_phase gen_memory --load_model 0
    python train.py --test_set $i --num_epochs 1000 --eta_min 1e-5  --batch_size 32 --learning_rate 1e-3  --randomRotate True  --phase test --train_phase train_adaptor --load_model 1000 --load_memory 500
done
# python train.py --test_set 3 --num_epochs 1000 --eta_min 1e-5  --batch_size 32 --learning_rate 1e-3  --randomRotate True  --phase train --train_phase train_diffusion --load_model 1000
