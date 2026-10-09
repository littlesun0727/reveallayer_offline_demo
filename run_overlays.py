#!/usr/bin/env python3
"""Offline RevealLayer experiment; does not modify the upstream checkout."""
import argparse
import ast
import gc
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import textwrap

# Set before importing any Hugging Face packages.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

from PIL import Image, ImageDraw, ImageOps

DEFAULT_TASK = '/data/w00911743/dataset/tasks/16-海边人像拼图'


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path('/data/w00911743/RevealLayer'))
    p.add_argument('--models', type=Path, default=Path('/data/w00911743/pretrained_models/reveallayer'))
    p.add_argument('--flux', type=Path, help='Explicit complete FLUX.1-dev directory')
    p.add_argument('--ckpt', type=Path, help='Directory containing RevealLayer LoRA and Refiner.pt')
    p.add_argument('--decoder', type=Path, help='Explicit transparent_decoder_ckpt.pth')
    p.add_argument('--image', type=Path, default=Path(DEFAULT_TASK) / 'prepared/reference.png')
    p.add_argument('--analysis', type=Path, default=Path(DEFAULT_TASK) / 'analysis.json')
    p.add_argument('--output', type=Path, default=Path(DEFAULT_TASK) / 'reveallayer_crop_s43')
    p.add_argument('--mode', choices=['crop', 'full'], default='crop')
    p.add_argument('--ids', nargs='+', help='Subset of overlay IDs; omitted means all')
    p.add_argument('--group-size', type=int, default=4, help='Targets per full-image pass, 1..10')
    p.add_argument('--context', type=float, default=0.35, help='Crop margin per side / bbox size')
    p.add_argument('--bbox-margin', type=int, default=0, help='Expand target bbox by N original-image pixels per side')
    p.add_argument('--min-crop', type=int, default=128)
    p.add_argument('--resolution', type=int, default=1024, help='Model input long side, multiple of 16')
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--guidance', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=43)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--offload', action='store_true', help='Accelerate model CPU offload for lower VRAM')
    p.add_argument('--prepare-only', action='store_true', help='Pillow + NumPy: inspect boxes without models/GPU')
    p.add_argument('--check-only', action='store_true', help='Prepare boxes and check local model files')
    a = p.parse_args()
    if not 1 <= a.group_size <= 10 or a.context < 0 or a.min_crop < 16 or a.bbox_margin < 0:
        p.error('group-size must be 1..10; context and bbox-margin >= 0; min-crop >= 16')
    if a.resolution < 256 or a.resolution % 16 or a.steps < 1:
        p.error('resolution must be >=256 and divisible by 16; steps >=1')
    return a


def load_targets(analysis, image_size, ids=None, margin=0):
    data = json.loads(Path(analysis).read_text(encoding='utf-8-sig'))
    rw, rh = data['reference_size']
    if rw <= 0 or rh <= 0:
        raise ValueError('Invalid reference_size')
    w, h = image_size
    rank = {name: i for i, name in enumerate(data.get('layer_order', []))}
    objects = [o for o in data['objects'] if o.get('kind') == 'overlay']
    objects.sort(key=lambda o: rank.get(o['id'], len(rank)))
    available = {o['id'] for o in objects}
    if len(available) != len(objects):
        raise ValueError('Duplicate overlay IDs')
    if ids and set(ids) - available:
        raise ValueError(f'Unknown overlay IDs: {set(ids) - available}')
    targets = []
    for o in objects:
        if ids and o['id'] not in ids:
            continue
        if not re.fullmatch(r'[A-Za-z0-9_-]+', o['id']):
            raise ValueError(f'Unsafe output ID: {o["id"]}')
        x1, y1, x2, y2 = o['bbox']
        box = [max(0, math.floor(x1*w/rw)), max(0, math.floor(y1*h/rh)),
               min(w, math.ceil(x2*w/rw)), min(h, math.ceil(y2*h/rh))]
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError(f'Empty bbox: {o["id"]}')
        expanded = [max(0, box[0]-margin), max(0, box[1]-margin),
                    min(w, box[2]+margin), min(h, box[3]+margin)]
        targets.append(dict(id=o['id'], label=o.get('label', ''), bbox=expanded, analysis_bbox=box))
    if not targets:
        raise ValueError('No overlay targets selected')
    return targets


def crop_box(box, size, context, min_crop):
    x1, y1, x2, y2 = box
    bw, bh = x2-x1, y2-y1
    cw, ch = max(min_crop, bw*(1+2*context)), max(min_crop, bh*(1+2*context))
    # Virtual crop may extend beyond image: edge-pad instead of losing context at borders.
    return [math.floor((x1+x2-cw)/2), math.floor((y1+y2-ch)/2),
            math.ceil((x1+x2+cw)/2), math.ceil((y1+y2+ch)/2)]


def extract_crop(image, box):
    import numpy as np
    x1, y1, x2, y2 = box
    w, h = image.size
    arr = np.asarray(image)
    arr = np.pad(arr, ((max(0, -y1), max(0, y2-h)),
                       (max(0, -x1), max(0, x2-w)), (0, 0)), mode='edge')
    return Image.fromarray(arr).crop((max(0, x1), max(0, y1),
                                      max(0, x1)+x2-x1, max(0, y1)+y2-y1))


def prepare_input(image, box, targets, resolution):
    crop = extract_crop(image, box)
    cw, ch = crop.size
    scale = resolution / max(cw, ch)
    rw, rh = max(1, round(cw*scale)), max(1, round(ch*scale))
    # Preserve the complete crop; pad to the 16-pixel latent grid (no center crop).
    mw, mh = math.ceil(rw/16)*16, math.ceil(rh/16)*16
    px, py = (mw-rw)//2, (mh-rh)//2
    resized = crop.resize((rw, rh), Image.Resampling.LANCZOS)
    model_image = extract_crop(resized, [-px, -py, mw-px, mh-py])
    boxes = []
    for t in targets:
        x1, y1, x2, y2 = t['bbox']
        q = [(x1-box[0])*rw/cw+px, (y1-box[1])*rh/ch+py,
             (x2-box[0])*rw/cw+px, (y2-box[1])*rh/ch+py]
        b = [max(0, math.floor(q[0]/16)*16), max(0, math.floor(q[1]/16)*16),
             min(mw, math.ceil(q[2]/16)*16), min(mh, math.ceil(q[3]/16)*16)]
        if b[0] >= b[2] or b[1] >= b[3]:
            raise ValueError(f'Empty transformed bbox: {t["id"]}')
        boxes.append(b)
    return model_image, dict(crop_box=box, resize_size=[rw, rh], pad=[px, py],
                             model_size=[mw, mh], boxes=boxes, targets=targets)


def restore_layer(layer, geometry, canvas_size):
    rw, rh = geometry['resize_size']
    px, py = geometry['pad']
    x1, y1, x2, y2 = geometry['crop_box']
    layer = layer.crop((px, py, px+rw, py+rh))
    # Pillow premultiplies RGBA for resampling, limiting dark edge halos.
    layer = layer.resize((x2-x1, y2-y1), Image.Resampling.LANCZOS)
    full = Image.new('RGBA', canvas_size)
    full.paste(layer, (x1, y1))  # no mask: preserve alpha without squaring it
    return full


def unique_file(root, filename):
    found = sorted(set(p.resolve() for p in root.rglob(filename) if p.is_file()))
    if len(found) != 1:
        raise ValueError(f'Expected one {filename} under {root}; found {found}. Use explicit path flags.')
    return found[0]


def model_paths(a):
    ckpt = a.ckpt.resolve() if a.ckpt else unique_file(a.models, 'pytorch_lora_weights.safetensors').parent
    flux = a.flux.resolve() if a.flux else unique_file(a.models, 'model_index.json').parent
    decoder = a.decoder.resolve() if a.decoder else unique_file(a.models, 'transparent_decoder_ckpt.pth')
    pe = ckpt / 'layer_pe_corrected.pt'
    if not pe.is_file():
        pe = ckpt / 'layer_pe.pt'
    required = [ckpt/'pytorch_lora_weights.safetensors', ckpt/'Refiner.pt', pe, decoder,
                flux/'model_index.json', flux/'scheduler/scheduler_config.json']
    for component in ['transformer', 'vae', 'text_encoder', 'text_encoder_2']:
        required.append(flux/component/'config.json')
        weights = list((flux/component).glob('*.safetensors')) + list((flux/component).glob('*.bin'))
        if not weights:
            raise FileNotFoundError(f'Missing model weights: {flux/component}')
        required.extend(weights)
        for index in (flux/component).glob('*.index.json'):
            mapping = json.loads(index.read_text())['weight_map']
            required.extend(flux/component/name for name in set(mapping.values()))
    for component in ['tokenizer', 'tokenizer_2']:
        folder = flux/component
        required.append(folder/'tokenizer_config.json')
        if not any((folder/name).is_file() for name in ['tokenizer.json', 'spiece.model', 'vocab.json']):
            raise FileNotFoundError(f'Missing tokenizer vocabulary: {folder}')
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f'Missing/empty file: {path}')
        with path.open('rb') as f:
            if f.read(100).startswith(b'version https://git-lfs.github.com/spec/'):
                raise ValueError(f'Git LFS pointer, not downloaded weights: {path}')
    return dict(flux=str(flux), ckpt=str(ckpt), decoder=str(decoder), pe=str(pe))


def decoder_constructor_without_bootstrap(init):
    """Remove only the upstream pretrained-ViT bootstrap, in memory, fail on code drift."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(init)))
    fn = tree.body[0]
    cutoff = next((i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'user_home' for t in n.targets)), None)
    if cutoff is None:
        raise RuntimeError('Upstream ViT constructor changed; review custom_model_xvae.py before running.')
    fn.body = fn.body[:cutoff]
    # Zero-argument super needs a class closure; compiling a standalone function lacks it.
    fn.body[0] = ast.parse('nn.Module.__init__(self)').body[0]
    ast.fix_missing_locations(tree)
    namespace = dict(init.__globals__)
    exec(compile(tree, '<offline-vit-init>', 'exec'), namespace)
    return namespace[fn.name]


def load_pipeline(a, paths):
    import torch
    from safetensors.torch import load_file
    repo = a.repo.resolve()
    if not (repo/'models/custom_pipeline.py').is_file() or not (repo/'diffusers/src/diffusers').is_dir():
        raise FileNotFoundError(f'Incomplete RevealLayer checkout: {repo}')
    sys.path[:0] = [str(repo/'diffusers/src'), str(repo)]
    import diffusers
    from diffusers import FluxTransformer2DModel, AutoencoderKL
    from models.custom_model_mmdit import CustomFluxTransformer2DModel
    from models.custom_pipeline import CustomFluxPipelineCfg
    from models import custom_model_xvae as xvae
    print('Using diffusers:', diffusers.__file__, flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this conda environment')
    torch.cuda.set_device(a.device)
    dtype = torch.bfloat16
    device = torch.device('cuda', a.device)
    print('GPU:', torch.cuda.get_device_name(device), flush=True)
    # Strict full decoder load guarantees the removed bootstrap leaves no random parameters.
    original_init = xvae.ViTEncoder.__init__
    xvae.ViTEncoder.__init__ = decoder_constructor_without_bootstrap(original_init)
    try:
        decoder = xvae.AutoencoderKLTransformerTraining()
    finally:
        xvae.ViTEncoder.__init__ = original_init
    state = torch.load(paths['decoder'], map_location='cpu', weights_only=True)
    decoder.load_state_dict(state, strict=True)
    del state
    decoder.eval().requires_grad_(False).to(device=device, dtype=dtype)
    print('Transparent decoder loaded completely.', flush=True)
    base = FluxTransformer2DModel.from_pretrained(paths['flux'], subfolder='transformer',
                                                 torch_dtype=dtype, local_files_only=True)
    config = dict(base.config)
    config['max_layer_num'] = 12
    transformer = CustomFluxTransformer2DModel.from_config(config).to(dtype=dtype)
    missing, extra = transformer.load_state_dict(base.state_dict(), strict=False)
    invalid = [k for k in missing if k != 'layer_pe' and not k.startswith('refiner.')]
    if invalid or extra:
        raise RuntimeError(f'Base transformer mismatch: {invalid}, {extra}')
    del base
    gc.collect()
    pe = torch.load(paths['pe'], map_location='cpu', weights_only=True)
    if pe.shape != transformer.layer_pe.shape:
        raise ValueError(f'layer_pe shape mismatch: {pe.shape} != {transformer.layer_pe.shape}')
    with torch.no_grad():
        transformer.layer_pe.copy_(pe)
    transformer.refiner.load_state_dict(torch.load(Path(paths['ckpt'])/'Refiner.pt',
                                                 map_location='cpu', weights_only=True), strict=True)
    vae = AutoencoderKL.from_pretrained(paths['flux'], subfolder='vae', torch_dtype=dtype, local_files_only=True)
    pipe = CustomFluxPipelineCfg.from_pretrained(paths['flux'], transformer=transformer,
                                                vae=vae, torch_dtype=dtype, local_files_only=True)
    lora = load_file(str(Path(paths['ckpt'])/'pytorch_lora_weights.safetensors'))
    lora = {k.replace('transformer.module.', 'transformer.'): v for k, v in lora.items()}
    pipe.load_lora_weights(lora, adapter_name='layer')
    pipe.set_adapters('layer', adapter_weights=1.0)
    del lora
    transformer.eval().requires_grad_(False)
    vae.eval().requires_grad_(False)
    vae.enable_slicing()
    if a.offload:
        pipe.enable_model_cpu_offload(gpu_id=a.device)
    else:
        pipe.to(device)
    return pipe, decoder, torch, device


def checker(size):
    image = Image.new('RGBA', size, '#eeeeee')
    draw = ImageDraw.Draw(image)
    for y in range(0, size[1], 12):
        for x in range(0, size[0], 12):
            if (x//12+y//12) % 2:
                draw.rectangle((x, y, x+11, y+11), fill='#bbbbbb')
    return image


def save_review(reference, targets, output):
    sheet = Image.new('RGB', (960, len(targets)*250), 'white')
    draw = ImageDraw.Draw(sheet)
    for row, t in enumerate(targets):
        folder = output/'elements'/t['id']
        rgba = Image.open(folder/'rgba.png').convert('RGBA')
        tiles = [reference.crop(t['bbox']), Image.alpha_composite(checker(rgba.size), rgba),
                 Image.alpha_composite(Image.new('RGBA', rgba.size, '#202020'), rgba)]
        for col, tile in enumerate(tiles):
            thumb = ImageOps.contain(tile.convert('RGB'), (310, 215))
            sheet.paste(thumb, (col*320+(320-thumb.width)//2, row*250+30))
        draw.text((8, row*250+6), t['id']+'   | reference / checker / dark', fill='black')
    sheet.save(output/'contact_sheet.jpg', quality=95)


def main():
    a = arguments()
    # Each experiment gets its own directory, so failed/old PNGs cannot masquerade as new results.
    if a.output.exists() and any(a.output.iterdir()):
        raise FileExistsError(f'Output is not empty; choose a new --output: {a.output}')
    a.output.mkdir(parents=True, exist_ok=True)
    reference = Image.open(a.image).convert('RGB')
    targets = load_targets(a.analysis, reference.size, a.ids, a.bbox_margin)
    overview = reference.copy()
    draw = ImageDraw.Draw(overview)
    for t in targets:
        draw.rectangle(t['bbox'], outline='red', width=2)
        draw.text(t['bbox'][:2], t['id'], fill='red', stroke_width=1, stroke_fill='white')
    overview.save(a.output/'boxes.png')
    groups = [[t] for t in targets] if a.mode == 'crop' else [targets[i:i+a.group_size] for i in range(0, len(targets), a.group_size)]
    jobs = []
    for i, group in enumerate(groups):
        box = crop_box(group[0]['bbox'], reference.size, a.context, a.min_crop) if a.mode == 'crop' else [0, 0, *reference.size]
        model_image, geometry = prepare_input(reference, box, group, a.resolution)
        folder = a.output/'passes'/f'{i:02d}'
        folder.mkdir(parents=True)
        model_image.save(folder/'input.png')
        marked = model_image.copy()
        d = ImageDraw.Draw(marked)
        for t, b in zip(group, geometry['boxes']):
            d.rectangle(b, outline='red', width=3)
            d.text(b[:2], t['id'], fill='red')
        marked.save(folder/'boxes.png')
        dump(folder/'geometry.json', geometry)
        jobs.append((folder, model_image, geometry))
    dump(a.output/'plan.json', dict(options={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                                   image_size=reference.size, targets=targets))
    print(f'Prepared {len(targets)} overlays in {len(jobs)} passes: {a.output}', flush=True)
    if a.prepare_only:
        return
    paths = model_paths(a)
    dump(a.output/'models.json', paths)
    print(json.dumps(paths, indent=2), flush=True)
    if a.check_only:
        return
    pipe, decoder, torch, device = load_pipeline(a, paths)
    import numpy as np
    for folder, model_image, geometry in jobs:
        print('Processing', folder.name, [t['id'] for t in geometry['targets']], flush=True)
        # Reset global RNG too: the upstream VAE samples without the explicit generator.
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
        np.random.seed(a.seed)
        w, h = model_image.size
        array = np.asarray(model_image).copy().astype(np.float32)/127.5-1
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        boxes = [(0, 0, w, h), (0, 0, w, h)] + [tuple(b) for b in geometry['boxes']]
        with torch.inference_mode():
            result = pipe(prompt='Decompose the image into foreground and background.',
                          full_image=tensor, validation_box=boxes, height=h, width=w,
                          num_layers=len(boxes), guidance_scale=a.guidance, num_inference_steps=a.steps,
                          generator=torch.Generator(device=device).manual_seed(a.seed), transparent_decoder=decoder)
        rgba = result[1]
        if rgba is None or rgba.shape != (len(boxes), h, w, 4):
            raise RuntimeError(f'Unexpected RGBA output shape: {getattr(rgba, "shape", None)}')
        Image.fromarray(rgba[0]).save(folder/'conditioning_reconstruction.png')
        Image.fromarray(rgba[1]).save(folder/'background_rgba.png')
        for j, t in enumerate(geometry['targets']):
            element = a.output/'elements'/t['id']
            element.mkdir(parents=True)
            raw = Image.fromarray(rgba[j+2])
            raw.save(element/'model_rgba.png')
            full = restore_layer(raw, geometry, reference.size)
            full.save(element/'full_rgba.png')
            tight = full.crop(t['bbox'])
            tight.save(element/'rgba.png')
            tight.getchannel('A').save(element/'alpha.png')
            # Diagnostic only: original RGB multiplied by generated alpha retains contamination.
            source_rgb = reference.crop(t['bbox']).convert('RGBA')
            source_rgb.putalpha(tight.getchannel('A'))
            source_rgb.save(element/'source_rgb_with_alpha.png')
            alpha = np.asarray(tight.getchannel('A'))
            dump(element/'metadata.json', dict(**t, pass_id=folder.name,
                 alpha_nonzero_fraction=float((alpha > 8).mean()),
                 alpha_opaque_fraction=float((alpha > 247).mean())))
        del result, rgba, tensor
        torch.cuda.empty_cache()
    save_review(reference, targets, a.output)
    dump(a.output/'DONE.json', {'status': 'complete', 'count': len(targets)})
    print('Done. Open:', a.output/'contact_sheet.jpg', flush=True)


if __name__ == '__main__':
    main()
