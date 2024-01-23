py -u run.py \
  --is_training 1 \
  --root_path ./dataset/MSL/ \
  --model_id KBJNet_MSL\
  --model KBJNet \
  --data MSL \
  --e_layers 2 \
  --anomaly_ratio 0.85 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 55 \
  --dec_in 55 \
  --c_out 55 \
  --n_windows 100\
  --gpu 0 \
  --p_hidden_dims 128 128 \
  --p_hidden_layers 2 \
  --itr 1 &

  # py -u run.py \
  # --is_training 0 \
  # --root_path ./dataset/MSL/ \
  # --model_id KBJNet_MSL\
  # --model KBJNet \
  # --data MSL \
  # --e_layers 2 \
  # --anomaly_ratio 1 \
  # --d_layers 1 \
  # --factor 3 \
  # --enc_in 55 \
  # --dec_in 55 \
  # --c_out 55 \
  # --n_windows 100\
  # --gpu 0 \
  # --p_hidden_dims 128 128 \
  # --p_hidden_layers 2 \
  # --itr 1 &