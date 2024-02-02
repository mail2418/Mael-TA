# py -u run_mantra.py \
#   --is_training 1 \
#   --root_path ./dataset/SWaT/ \
#   --model_id MaelNetB1_MaelNetS1_SWaT_Negative_Corr \
#   --model MaelNetB1 \
#   --slow_model MaelNetS1 \
#   --data SWaT \
#   --e_layers 2 \
#   --d_layers 1 \
#   --anomaly_ratio 1 \
#   --factor 5 \
#   --enc_in 51 \
#   --dec_in 51 \
#   --c_out 51 \
#   --d_model 51 \
#   --gpu 0 \
#   --p_hidden_dims 128 128 \
#   --p_hidden_layers 2 \
#   --itr 1 &

  py -u run_mantra.py \
  --is_training 0 \
  --root_path ./dataset/SWaT/ \
  --model_id MaelNetB1_MaelNetS1_SWaT_Negative_Corr \
  --model MaelNetB1 \
  --slow_model MaelNetS1 \
  --data SWaT \
  --e_layers 2 \
  --d_layers 1 \
  --anomaly_ratio 1 \
  --factor 5 \
  --enc_in 51 \
  --dec_in 51 \
  --c_out 51 \
  --d_model 51 \
  --gpu 0 \
  --p_hidden_dims 128 128 \
  --p_hidden_layers 2 \
  --itr 1 &