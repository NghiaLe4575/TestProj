from clip import _clip_ebc

# Test EfficientNetV2-S
model = _clip_ebc(
    backbone="efficientnet_v2_s",
    bins=[(0, 10), (11, 50), (51, 100)],  # Dummy bins for testing
    anchor_points=[5, 30, 75],  
    reduction=8
)

print(f"Model initialized with EfficientNetV2-S!")
print(f"Backbone: {model.backbone}")
print(f"Decoder Block: {model.decoder_block}")
print(f"Decoder Config: {model.decoder_cfg}")
