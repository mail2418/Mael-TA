# MaelNet
py -u run_anomaly.py \
  --is_training 1 \
  --root_path ./dataset/SMAP/ \
  --model_id MaelNetS1_AnomalyTransformer_DCDetector_RL\
  --model MaelNetS2 \
  --is_slow_learner true \
  --data SMAP \
  --e_layers 2 \
  --d_layers 1 \
  --anomaly_ratio 0.85 \
  --factor 5 \
  --enc_in 25 \
  --dec_in 25 \
  --c_out 25 \
  --d_model 512 \
  --moving_avg 100 \
  --gpu 0 \
  --des 'Exp_h256_l2' \
  --p_hidden_dims 128 128 \
  --p_hidden_layers 2 \
  --itr 1 &

# DCDetector
py -u run_anomaly.py \
  --is_training 1 \
  --root_path ./dataset/SMAP/ \
  --model_id MaelNetS1_AnomalyTransformer_DCDetector_RL\
  --model DCDetector \
  --patch_size 25 \
  --train_epochs 3 \
  --data SMAP \
  --e_layers 2 \
  --d_layers 1 \
  --anomaly_ratio 0.85 \
  --factor 5 \
  --enc_in 25 \
  --dec_in 25 \
  --c_out 25 \
  --d_model 512 \
  --moving_avg 100 \
  --gpu 0 \
  --des 'Exp_h256_l2' \
  --p_hidden_dims 128 128 \
  --p_hidden_layers 2 \
  --itr 1 &

# Anomaly Transformer
py -u run_anomaly.py \
  --is_training 1 \
  --root_path ./dataset/SMAP/ \
  --model_id MaelNetS1_AnomalyTransformer_DCDetector_RL\
  --model AnomalyTransformer \
  --train_epochs 3 \
  --data SMAP \
  --e_layers 2 \
  --d_layers 1 \
  --anomaly_ratio 0.85 \
  --factor 5 \
  --enc_in 25 \
  --dec_in 25 \
  --c_out 25 \
  --d_model 512 \
  --moving_avg 100 \
  --gpu 0 \
  --des 'Exp_h256_l2' \
  --p_hidden_dims 128 128 \
  --p_hidden_layers 2 \
  --itr 1 &