"""Make balanced train calibration and disjoint validation inputs from Imagenette2-160."""
import argparse
import hashlib
import io
import json
import random
import tarfile
from pathlib import Path

from PIL import Image

LABELS={'n01440764':0,'n02102040':217,'n02979186':482,'n03000684':491,'n03028079':497,
        'n03394916':566,'n03417042':569,'n03425413':571,'n03445777':574,'n03888257':701}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',type=Path)
    p.add_argument('out',type=Path)
    args=p.parse_args()
    args.out.mkdir(exist_ok=False)
    rng=random.Random(49621)
    manifest=[]
    with tarfile.open(args.archive) as archive:
        images=[m for m in archive if m.isfile() and m.name.lower().endswith('.jpeg')]
        selection=[]
        for split in ('train','val'):
            (args.out/split).mkdir()
            for i,cls in enumerate(sorted(LABELS)):
                choices=sorted([m for m in images if '/{}/{}/'.format(split,cls) in m.name],key=lambda m:m.name)
                count=(13 if i<8 else 12) if split=='train' else 4
                selection.extend((m,cls,split,j) for j,m in enumerate(rng.sample(choices,count)))
        for member,cls,split,j in sorted(selection,key=lambda s:s[0].offset_data):
            image=Image.open(io.BytesIO(archive.extractfile(member).read())).convert('RGB')
            w,h=image.size
            size=(256,256*h//w) if w<=h else (256*w//h,256)
            image=image.resize(size,Image.Resampling.BICUBIC)
            left,top=round((size[0]-224)/2),round((size[1]-224)/2)
            path=args.out/split/('{}-{:02d}.png'.format(cls,j))
            image.crop((left,top,left+224,top+224)).save(path)
            manifest.append(dict(source=member.name,file=str(path.relative_to(args.out)),class_index=LABELS[cls],
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (args.out/'manifest.json').write_text(json.dumps(dict(seed=49621,archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),images=manifest),indent=2))
if __name__=='__main__': main()
