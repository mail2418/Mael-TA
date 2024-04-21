# py -u run_mantra_rl.py \
#   --is_training 1 \
#   --root_path ./dataset/MSL/ \
#   --model_id MaelNetB1_MaelNetS1_MSL_Negative_Corr_RL_2_epoch10_itr1500 \
#   --model MaelNetB1 \
#   --slow_model MaelNetS1 \
#   --data MSL \
#   --e_layers 2 \
#   --d_layers 1 \
#   --anomaly_ratio 1 \
#   --factor 5 \
#   --enc_in 55 \
#   --dec_in 55 \
#   --c_out 55 \
#   --d_model 55 \
#   --gpu 0 \
#   --p_hidden_dims 128 128 \
#   --p_hidden_layers 2 \
#   --epoch_itr 1500 \
#   --itr 1 &

  py -u run_mantra_rl2.py \
  --is_training 1 \
  --root_path ./dataset/MSL/ \
  --model_id MaelNetB1_MaelNetS1_MSL_Negative_Corr_RL_2_epoch10_itr1500 \
  --model MaelNetB1 \
  --slow_model MaelNetS1 \
  --data MSL \
  --e_layers 2 \
  --d_layers 1 \
  --anomaly_ratio 1 \
  --factor 5 \
  --enc_in 55 \
  --dec_in 55 \
  --c_out 55 \
  --d_model 55 \
  --gpu 0 \
  --p_hidden_dims 128 128 \
  --p_hidden_layers 2 \
  --epoch_itr 1500 \
  --itr 1 &