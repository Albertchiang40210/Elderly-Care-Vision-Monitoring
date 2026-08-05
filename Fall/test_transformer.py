import torch
import json
from modules.action_transformer import ActionTransformer
from mlops_config.settings import settings
import numpy as np

with open("models/action_classifier/label_map.json", "r") as f:
    label_map = json.load(f)
id_to_label = {int(k): v for k, v in label_map.items()}

model = ActionTransformer(
    num_classes=len(label_map),
    input_dim=settings.INPUT_DIM,
    d_model=settings.D_MODEL,
    nhead=settings.NHEAD,
    num_layers=settings.NUM_LAYERS
)
model.load_state_dict(torch.load("models/action_classifier/transformer_action_model.pt", map_location='cpu'))
model.eval()

# Chaotic sequence
seq = np.random.rand(1, 30, 34).astype(np.float32)
seq_tensor = torch.from_numpy(seq).cpu()

with torch.no_grad():
    logits = model(seq_tensor)
    probs = torch.softmax(logits, dim=-1)
    
print("Probabilities for chaotic sequence:")
for i in range(len(label_map)):
    print(f"{id_to_label[i]}: {probs[0, i].item():.4f}")
