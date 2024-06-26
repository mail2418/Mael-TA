py -u run.py \
  --is_training 1 \
  --root_path ./dataset/SWaT/ \
  --model_id NSTransformer_SWaT \
  --model NSTransformer \
  --data SWaT \
  --e_layers 2 \
  --anomaly_ratio 0.6 \
  --d_layers 1 \
  --factor 5 \
  --enc_in 51 \
  --dec_in 51 \
  --c_out 51 \
  --d_model 512 \
  --gpu 0 \
  --des 'Exp_h256_l2' \
  --p_hidden_dims 128 128 \
  --p_hidden_layers 2 \
  --itr 1 &