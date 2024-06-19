# MaelNet
# python -u run_anomaly.py \
#   --is_training 1 \
#   --root_path ./dataset/MSL/ \
#   --model_id MaelNetS2_AnomalyTransformer_DCDetector_RL\
#   --model MaelNetS2 \
#   --is_slow_learner true \
#   --data MSL \
#   --e_layers 2 \
#   --d_layers 1 \
#   --anomaly_ratio 0.85 \
#   --factor 5 \
#   --enc_in 55 \
#   --dec_in 55 \
#   --c_out 55 \
#   --d_model 512 \
#   --moving_avg 100 \
#   --gpu 0 \
#   --des 'Exp_h256_l2' \
#   --p_hidden_dims 128 128 \
#   --p_hidden_layers 2 \
#   --itr 1 &

# DCDetector
# python -u run_anomaly.py \
#   --is_training 1 \
#   --root_path ./dataset/MSL/ \
#   --model_id MaelNetS2_AnomalyTransformer_DCDetector_RL\
#   --model DCDetector \
#   --patch_size 5 \
#   --train_epochs 3 \
#   --data MSL \
#   --e_layers 2 \
#   --d_layers 1 \
#   --anomaly_ratio 0.85 \
#   --factor 5 \
#   --enc_in 55 \
#   --dec_in 55 \
#   --c_out 55 \
#   --channel 55 \
#   --d_model 512 \
#   --moving_avg 100 \
#   --win_size 100 \
#   --gpu 0 \
#   --des 'Exp_h256_l2' \
#   --p_hidden_dims 128 128 \
#   --p_hidden_layers 2 \
#   --itr 1 &

# Anomaly Transformer
python -u run_anomaly.py \
  --is_training 1 \
  --root_path ./dataset/MSL/ \
  --model_id MaelNetS2_AnomalyTransformer_DCDetector_RL\
  --model AnomalyTransformer \
  --train_epochs 3 \
  --data MSL \
  --e_layers 2 \
  --d_layers 1 \
  --anomaly_ratio 0.85 \
  --factor 5 \
  --enc_in 55 \
  --dec_in 55 \
  --c_out 55 \
  --d_model 512 \
  --moving_avg 100 \
  --win_size 100 \
  --gpu 0 \
  --des 'Exp_h256_l2' \
  --p_hidden_dims 128 128 \
  --p_hidden_layers 2 \
  --itr 1 &