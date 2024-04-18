py -u run_mantra_rl.py \
  --is_training 1 \
  --root_path ./dataset/SMAP/ \
  --model_id NSTransformerB1_NSTransformerS1_SMAP_Negative_Corr_RL_1_epoch10_itr1500 \
  --model NSTransformerB1 \
  --slow_model NSTransformerS1 \
  --data SMAP \
  --e_layers 2 \
  --d_layers 1 \
  --anomaly_ratio 1 \
  --factor 5 \
  --enc_in 25 \
  --dec_in 25 \
  --c_out 25 \
  --d_model 512 \
  --gpu 0 \
  --p_hidden_dims 128 128 \
  --p_hidden_layers 2 \
  --epoch_itr 1500 \
  --itr 1 &