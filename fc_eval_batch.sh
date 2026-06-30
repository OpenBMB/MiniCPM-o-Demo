python evaluate_fc_duplex_batch.py \
  --ref-audio-path /user/heweiquan/dataset/DuplexFcTest/delivery_train_data/media/system_reference/HTRef06.wav \
  --tts-prompt-path /user/heweiquan/dataset/DuplexFcTest/delivery_train_data/media/system_reference/HTRef06.wav \
  --attn-implementation sdpa \
  --decode-mode greedy \
  --skip-mutated