#!/usr/bin/env python3
"""Replay the same saved frames in each 21M mode on HPS. CSV times are milliseconds."""
import argparse
import csv
import hashlib
from pathlib import Path

from PIL import Image
from hps_local_demo import Backend, preprocess, verify_model

def main():
    folder=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images',type=Path)
    parser.add_argument('csv',type=Path)
    parser.add_argument('--modes',nargs='+',default=['cpu-fp32','cpu-mixed','hybrid-unfused','hybrid'],
                        choices=['cpu-fp32','cpu-mixed','hybrid-unfused','hybrid'])
    parser.add_argument('--passes',default=2,type=int,help='Odd passes run modes in reverse order')
    parser.add_argument('--threads',type=int,choices=[1,2],default=2)
    parser.add_argument('--warmup',type=int,default=3)
    args=parser.parse_args()
    images=sorted(p for p in args.images.iterdir() if p.suffix.lower() in ('.jpg','.png','.jpeg','.ppm'))
    if not images or args.passes<=0 or args.warmup<0: parser.error('Provide images and valid counts')
    args.runner=str(folder/'bin'/'tinyvit_stream')
    args.timeout,args.profile=120,False
    fields=['pass','mode','threads','image','tensor_sha256','model_sha256','session_setup_ms','infer_ms',
            'runner_peak_kib','jobs','layers','pairs','top1']
    def load_image(p):
        with Image.open(str(p)) as im: return preprocess(im)[1]
    with args.csv.open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(fields)
        for block in range(args.passes):
            for mode in (args.modes if block%2==0 else reversed(args.modes)):
                args.mode,args.model=mode,None
                verify_model(args,folder)
                backend=Backend(args)
                try:
                    for i in range(args.warmup):
                        backend.predict(i+1,load_image(images[i%len(images)]))
                    for i,p in enumerate(images):
                        raw=load_image(p)
                        y,m=backend.predict(args.warmup+i+1,raw)
                        writer.writerow([block+1,mode,args.threads,p.name,hashlib.sha256(raw).hexdigest(),
                                         args.model_hash,backend.load_ms,m['infer_ms'],m['runner_peak_kib'],
                                         m['jobs'],m['layers'],m.get('native',{}).get('fused_pairs',0),int(y.argmax())])
                        f.flush()
                finally: backend.close()
if __name__=='__main__': main()
