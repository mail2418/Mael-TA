py -u run.py \
  --is_training 1 \
  --root_path ./dataset/PSM/ \
  --model_id NSTransformer_PSM \
  --model NSTransformer \
  --data PSM \
  --e_layers 2 \
  --d_layers 1 \
  --anomaly_ratio 0.85 \
  --factor 5 \
  --enc_in 25 \
  --dec_in 25 \
  --c_out 25 \
  --d_model 512 \
  --gpu 0 \
  --des 'Exp_h256_l2' \
  --p_hidden_dims 128 128 \
  --p_hidden_layers 2 \
  --itr 1 &

  # py -u run.py \
  # --is_training 0 \
  # --root_path ./dataset/PSM/ \
  # --model_id NSTransformer_PSM \
  # --model NSTransformer \
  # --data PSM \
  # --e_layers 2 \
  # --d_layers 1 \
  # --anomaly_ratio 0.85 \
  # --factor 5 \
  # --enc_in 25 \
  # --dec_in 25 \
  # --c_out 25 \
  # --d_model 512 \
  # --gpu 0 \
  # --des 'Exp_h256_l2' \
  # --p_hidden_dims 128 128 \
  # --p_hidden_layers 2 \
  # --itr 1 &