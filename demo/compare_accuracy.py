"""Score two FP32 MNN classifiers on the same labeled HPS inputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from hps_local_demo import Backend, preprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='New per-image results CSV')
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output filename")
    root = Path(__file__).resolve().parent
    dataset = root / 'accuracy200'
    manifest = json.loads((dataset / 'manifest.json').read_text())
    images = manifest['images']
    assert len(images) == 200 and len(set(row['sha256'] for row in images)) == 200
    results = {}
    hashes = {}
    for size in ('5m', '21m'):
        model = root / 'models' / ('tinyvit-' + size + '-fp32.mnn')
        if hashlib.sha256(model.read_bytes()).hexdigest() != manifest['models'][model.name]:
            raise ValueError('Unexpected model: ' + model.name)
        backend = Backend(argparse.Namespace(mode='cpu-fp32', profile=False, timeout=120,
                          threads=2, model=str(model), runner=str(root / 'bin' / 'tinyvit_stream')))
        predictions = []
        try:
            for i, row in enumerate(images):
                path = dataset / 'images' / row['file']
                if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                    raise ValueError('Unexpected image: ' + str(path))
                with Image.open(str(path)) as image:
                    _, raw = preprocess(image, 'camera-crop')
                digest = hashlib.sha256(raw).hexdigest()
                if hashes.setdefault(row['file'], digest) != digest:
                    raise ValueError('Input tensor changed between models')
                scores, _ = backend.predict(i + 1, raw)
                predictions.append(np.argsort(-scores, kind='mergesort')[:5].tolist())
                if (i + 1) % 20 == 0:
                    print('{}: {}/200'.format(size, i + 1), flush=True)
        finally:
            backend.close()
        results[size] = predictions
    with args.output.open('x', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['image','imagenet_label','tensor_sha256','5m_ranked_top5','21m_ranked_top5'])
        for i, row in enumerate(images):
            writer.writerow([row['file'], row['label'], hashes[row['file']],
                             results['5m'][i], results['21m'][i]])
    for size in ('5m','21m'):
        for k in (1,5):
            correct = sum(row['label'] in pred[:k] for row,pred in zip(images,results[size]))
            print('{} top-{}: {}/200 ({:.1f}%)'.format(size,k,correct,correct/2.0))


if __name__ == '__main__':
    main()
