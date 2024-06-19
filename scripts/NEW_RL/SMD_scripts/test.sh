py -u run_anomaly.py \
  --is_training 0 \
<<<<<<< HEAD
  --root_path ./dataset/SMAP/ \
  --model_id MaelNetS2_AnomalyTransformer_DCDetector_RL\
=======
  --root_path ./dataset/SMD/ \
  --model_id MaelNetS2_AnomalyTransformer_DCDetector_RL_TA\
>>>>>>> 18c16565f4d42a9d914a90cc95188660d63d2440
  --data SMD \
  --win_size 100 \
  --channel 38 \
  --patch_size 5 \
  --e_layers 3 \
  --d_layers 1 \
  --anomaly_ratio 0.6 \
  --factor 5 \
  --enc_in 38 \
  --dec_in 38 \
  --c_out 38 \
  --d_ff 512 \
  --dropout 0.0 \
  --d_model 512 \
  --moving_avg 100 \
  --gpu 0 \
  --des 'TA' \
  --p_hidden_dims 128 128 \
  --p_hidden_layers 2 \
  --itr 1 &
