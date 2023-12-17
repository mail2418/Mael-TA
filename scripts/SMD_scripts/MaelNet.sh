python -u run.py \
  --is_training 1 \
  --root_path ./dataset/SMD/ \
  --model_id MaelNet_100_96 \
  --model MaelNet \
  --data SMD \
  --e_layers 2 \
  --anomaly_ratio 0.6 \
  --d_layers 1 \
  --factor 5 \
  --enc_in 38 \
  --dec_in 38 \
  --c_out 38 \
  --d_model 38 \
  --moving_avg 100 \
  --gpu 0 \
  --des 'Exp_h256_l2' \
  --p_hidden_dims 128 128 \
  --p_hidden_layers 2 \
  --itr 1 &