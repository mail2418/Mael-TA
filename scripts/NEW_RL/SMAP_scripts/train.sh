# MaelNet
export CUDA_VISIBLE_DEVICES=0

# py -u run_anomaly.py \
#   --is_training 1 \
#   --root_path ./dataset/SMAP/ \
#   --model_id MaelNetS1_AnomalyTransformer_DCDetector_RL3\
#   --model MaelNetS2 \
#   --is_slow_learner true \
#   --data SMAP \
#   --e_layers 3 \
#   --d_layers 1 \
#   --anomaly_ratio 0.85 \
#   --factor 5 \
#   --enc_in 25 \
#   --dec_in 25 \
#   --c_out 25 \
#   --d_model 256 \
#   --moving_avg 105 \
#   --win_size 105 \
#   --gpu 0 \
#   --des 'Exp_h256_l2' \
#   --p_hidden_dims 128 128 \
#   --p_hidden_layers 2 \
#   --itr 1 &

# DCDetector
py -u run_anomaly.py \
  --is_training 1 \
  --root_path ./dataset/SMAP/ \
  --model_id MaelNetS1_AnomalyTransformer_DCDetector_RL3\
  --model DCDetector \
  --patch_size 357 \
  --train_epochs 3 \
  --data SMAP \
  --e_layers 3 \
  --d_layers 1 \
  --anomaly_ratio 0.85 \
  --factor 5 \
  --channel 25 \
  --enc_in 25 \
  --dec_in 25 \
  --c_out 25 \
  --d_model 256 \
  --moving_avg 105 \
  --win_size 105 \
  --gpu 0 \
  --des 'Exp_h256_l2' \
  --p_hidden_dims 128 128 \
  --p_hidden_layers 2 \
  --itr 1 &

# # Anomaly Transformer
# py -u run_anomaly.py \
#   --is_training 1 \
#   --root_path ./dataset/SMAP/ \
#   --model_id MaelNetS1_AnomalyTransformer_DCDetector_RL3\
#   --model AnomalyTransformer \
#   --patch_size 5 \
#   --train_epochs 3 \
#   --data SMAP \
#   --e_layers 3 \
#   --d_layers 1 \
#   --anomaly_ratio 0.85 \
#   --factor 5 \
#   --enc_in 25 \
#   --dec_in 25 \
#   --c_out 25 \
#   --d_model 256 \
#   --moving_avg 105 \
#   --win_size 105 \
#   --gpu 0 \
#   --des 'Exp_h256_l2' \
#   --p_hidden_dims 128 128 \
#   --p_hidden_layers 2 \
#   --itr 1 &