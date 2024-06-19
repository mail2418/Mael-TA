export CUDA_VISIBLE_DEVICES=0

py -u run_anomaly.py \
  --is_training 0 \
  --root_path ./dataset/PSM/ \
  --model_id MaelNetS2_AnomalyTransformer_DCDetector_RL_TA\
  --patch_size 5 \
  --train_epochs 3 \
  --data PSM \
  --e_layers 3 \
  --d_layers 1 \
  --anomaly_ratio 0.85 \
  --factor 5 \
  --d_ff 512 \
  --dropout 0.0 \
  --enc_in 25 \
  --dec_in 25 \
  --channel 25 \
  --c_out 25 \
  --d_model 512 \
  --moving_avg 100 \
  --win_size 100 \
  --gpu 0 \
  --des 'TA' \
  --p_hidden_dims 128 128 \
  --p_hidden_layers 2 \
  --itr 1 &