# MaelNet
# py -u run_anomaly.py \
#   --is_training 1 \
#   --root_path ./dataset/SMD/ \
#   --model_id MaelNetS2_AnomalyTransformer_DCDetector_RL2\
#   --model MaelNetS2 \
#   --is_slow_learner true \
#   --data SMD \
#   --e_layers 3 \
#   --d_layers 1 \
#   --anomaly_ratio 0.6 \
#   --factor 5 \
#   --enc_in 38 \
#   --dec_in 38 \
#   --c_out 38 \
#   --d_model 256 \
#   --moving_avg 105 \
#   --win_size 105 \
#   --gpu 0 \
#   --des 'Exp_h256_l2' \
#   --p_hidden_dims 128 128 \
#   --p_hidden_layers 2 \
#   --itr 1 &

# # DCDetector
# py -u run_anomaly.py \
#   --is_training 1 \
#   --root_path ./dataset/SMD/ \
#   --model_id MaelNetS2_AnomalyTransformer_DCDetector_RL2\
#   --model DCDetector \
#   --patch_size 57 \
#   --train_epochs 3 \
#   --data SMD \
#   --e_layers 3 \
#   --d_layers 1 \
#   --anomaly_ratio 0.6 \
#   --factor 5 \
#   --enc_in 38 \
#   --dec_in 38 \
#   --channel 38 \
#   --c_out 38 \
#   --d_model 256 \
#   --moving_avg 105 \
#   --win_size 105 \
#   --gpu 0 \
#   --des 'Exp_h256_l2' \
#   --p_hidden_dims 128 128 \
#   --p_hidden_layers 2 \
#   --itr 1 &

# # Anomaly Transformer
py -u run_anomaly.py \
  --is_training 1 \
  --root_path ./dataset/SMD/ \
  --model_id MaelNetS2_AnomalyTransformer_DCDetector_RL2\
  --model AnomalyTransformer \
  --train_epochs 3 \
  --data SMD \
  --e_layers 3 \
  --d_layers 1 \
  --anomaly_ratio 0.6 \
  --factor 5 \
  --enc_in 38 \
  --dec_in 38 \
  --c_out 38 \
  --d_model 256 \
  --moving_avg 105 \
  --win_size 105 \
  --gpu 0 \
  --des 'Exp_h256_l2' \
  --p_hidden_dims 128 128 \
  --p_hidden_layers 2 \
  --itr 1 &