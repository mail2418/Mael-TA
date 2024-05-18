# py -u run_mantra_rl.py \
#   --is_training 1 \
#   --root_path ./dataset/SMD/ \
#   --model_id MaelNetB1_MaelNetS1_SMD_Negative_Corr_RL_1_epoch10 \
#   --model MaelNetB1 \
#   --slow_model MaelNetS1 \
#   --data SMD \
#   --e_layers 2 \
#   --d_layers 1 \
#   --anomaly_ratio 0.5 \
#   --factor 5 \
#   --enc_in 38 \
#   --dec_in 38 \
#   --c_out 38 \
#   --d_model 38 \
#   --gpu 0 \
#   --p_hidden_dims 128 128 \
#   --p_hidden_layers 2 \
#   --epoch_itr 1500 \
#   --itr 1 &

python -u run_mantra_rl2.py \
  --is_training 1 \
  --root_path ./dataset/SMD/ \
  --model_id MaelNetB1_MaelNetS1_SMD_Negative_Corr_RL_1_epoch10 \
  --model MaelNetB1 \
  --slow_model MaelNetS1 \
  --data SMD \
  --e_layers 2 \
  --d_layers 1 \
  --anomaly_ratio 0.5 \
  --factor 5 \
  --enc_in 38 \
  --dec_in 38 \
  --c_out 38 \
  --d_model 38 \
  --gpu 0 \
  --p_hidden_dims 128 128 \
  --p_hidden_layers 2 \
  --epoch_itr 1500 \
  --itr 1 &