"""Export the team's exact TinyViT-21M checkpoint as a fixed batch-1 FP32 ONNX."""
import argparse
import hashlib
import sys
from pathlib import Path

import torch
from model21.tiny_vit import tiny_vit_21m_224

CHECKPOINT = 'eb20032633663fc3e1e49c1cb21332c11e6804c9f920b77da5f832dad248b897'

def load(checkpoint):
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != CHECKPOINT:
        raise ValueError('Expected official 21M / 224 / 22k-to-1k distillation checkpoint')
    model = tiny_vit_21m_224(pretrained=False)
    state = torch.load(str(checkpoint),weights_only=True,map_location='cpu')['model']
    model.load_state_dict({k:v for k,v in state.items() if not k.endswith('attention_bias_idxs')},strict=True)
    return model.float().eval()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = load(args.checkpoint)
    with torch.no_grad():
        torch.onnx.export(model, torch.zeros(1,3,224,224), str(args.output), opset_version=17,
                          input_names=['image'], output_names=['logits'], dynamo=False)
    print('TinyViT-21M · {} parameters · {}'.format(sum(p.numel() for p in model.parameters()),args.output))
