"""Check the exported FP32 and mixed MNN graphs on identical pre-cropped validation PNGs."""
import argparse
import json
from pathlib import Path
import MNN
import numpy as np
from PIL import Image

def open_model(path):
    net=MNN.Interpreter(str(path))
    net.setSessionMode(7)
    session=net.createSession({'backend':'CPU','numThread':2,'precision':'high'})
    inp=net.getSessionInput(session)
    net.resizeTensor(inp,(1,3,224,224))
    net.resizeSession(session)
    return net,session,inp

def evaluate(instance,x):
    net,session,inp=instance
    inp.copyFrom(MNN.Tensor(x.shape,MNN.Halide_Type_Float,x,MNN.Tensor_DimensionType_Caffe))
    net.runSession(session)
    out=net.getSessionOutput(session)
    host=MNN.Tensor(out.getShape(),MNN.Halide_Type_Float,np.zeros(out.getShape(),np.float32),out.getDimensionType())
    out.copyToHostTensor(host)
    scores=host.getNumpyData().reshape(-1).copy()
    if scores.size!=1000 or not np.isfinite(scores).all(): raise ValueError('Invalid scores')
    return scores

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('models',type=Path)
    parser.add_argument('data',type=Path)
    parser.add_argument('report',type=Path)
    parser.add_argument('--pytorch-logits',type=Path)
    args=parser.parse_args()
    fp,mix=[open_model(args.models/f) for f in ('tinyvit-21m-fp32.mnn','tinyvit-21m-mlp-w8a8.mnn')]
    rows=[]
    for item in json.loads((args.data/'manifest.json').read_text())['images']:
        if not item['file'].startswith('val/'): continue
        p=args.data/item['file']
        pixels=np.asarray(Image.open(p).convert('RGB'),dtype=np.float32)/255
        data=(pixels-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        x=np.ascontiguousarray(data.transpose(2,0,1)[None])
        a,b=evaluate(fp,x),evaluate(mix,x)
        label=item['class_index']
        row=dict(image=p.name,label=label,fp32_top1=int(a.argmax()),mixed_top1=int(b.argmax()),
                 fp32_top5=list(map(int,np.argsort(-a)[:5])),mixed_top5=list(map(int,np.argsort(-b)[:5])),
                 cosine=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))))
        if args.pytorch_logits:
            pt=np.fromfile(str(args.pytorch_logits/(p.stem+'.f32')),'<f4')
            row.update(torch_top1=int(pt.argmax()),torch_fp32_max_abs=float(abs(pt-a).max()))
        rows.append(row)
    summary=dict(n=len(rows),fp32_correct=sum(r['label']==r['fp32_top1'] for r in rows),
                 mixed_correct=sum(r['label']==r['mixed_top1'] for r in rows),
                 fp32_top5_correct=sum(r['label'] in r['fp32_top5'] for r in rows),
                 mixed_top5_correct=sum(r['label'] in r['mixed_top5'] for r in rows),
                 top1_agreement=sum(r['fp32_top1']==r['mixed_top1'] for r in rows),
                 mean_cosine=float(np.mean([r['cosine'] for r in rows])))
    if args.pytorch_logits:
        summary.update(torch_fp32_top1_agreement=sum(r['torch_top1']==r['fp32_top1'] for r in rows),
                       torch_fp32_max_abs=max(r['torch_fp32_max_abs'] for r in rows))
    args.report.write_text(json.dumps(dict(platform='Host CPU, MNN 3.6.2',summary=summary,images=rows),indent=2)+'\n')
    print(summary)
if __name__=='__main__': main()
