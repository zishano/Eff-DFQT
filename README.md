# Eff-DFQT
We plan to make the source code related to this paper public in the near future. Please continue to follow this project for the latest information and updates. Thank you for your patience and support!
![img10](https://github.com/user-attachments/assets/5500b323-5a2d-4ac6-bfa7-451ef45b8d2b)

# Requirements
One high-end GPU for inference such as an RTX 3090
* Install [PyTorch](http://pytorch.org/)
* pip install -r requirements.txt

# Model Quantization
  - Example: Quantize (W8/A8) DeiT/16-Base with inverted data (Eff-DFQT).
```
python test_quant_tome_test.py --model deit_tiny_16_imagenet \
    --prune_it 50 100 200 300 \
    --prune_ratio 0.3 0.3 0.3 0.3 \
    --dataset ../data/imagenet \
    --datapool ./output \
    --mode 0 \
    --w_bit 8 --a_bit 8 \
    --calib-batchsize 128 \
    --val-batchsize 50 \
    --gpu 0 \
    --ldr 0 \
    --ratio 2 1 2 1 2 1 2 1 2 1 2 1
```

# Result 
![image](https://github.com/user-attachments/assets/c1609a83-df86-41cd-a519-53703c6d431b)
