"""Bounded, foreground CST-FFN learning run; provenance and raw curves retained."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .ffn_contract import ModelConfig
from .ffn_model import LanguageModel, configure_native, cst_optimizer


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def token_batch(tokens, starts, context, device):
    """Shift targets once; no circular wrap and no cross-split window reads."""
    rows = np.stack([tokens[int(i):int(i) + context + 1] for i in starts]).astype(np.int64)
    if rows.shape[1] != context + 1:
        raise ValueError('next-token window exceeds its split')
    data = torch.from_numpy(rows).to(device)
    return data[:, :-1], data[:, 1:]


def verify_data(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['state'] != 'prepared' or not manifest['story_hash_sets_disjoint']:
        raise ValueError('prepared story-disjoint token data required')
    result = {}
    for name, entry in manifest['files'].items():
        path = root / entry['filename']
        if sha(path) != entry['sha256'] or path.stat().st_size != 4 * entry['tokens']:
            raise ValueError('token file changed: ' + name)
        result[name] = np.memmap(path, mode='r', dtype='<u4')
        if int(result[name].max()) >= manifest['vocabulary']:
            raise ValueError('token outside vocabulary')
    return manifest, result


@torch.no_grad()
def evaluate(model, tokens, batch):
    model.eval()
    length = model.config.context
    windows = (len(tokens) - 1) // length
    total, count = 0.0, 0
    for first in range(0, windows, batch):
        starts = np.arange(first, min(windows, first + batch)) * length
        x, y = token_batch(tokens, starts, length, 'cuda')
        _, loss = model(x, y)
        total += float(loss) * x.numel(); count += x.numel()
    return {'ce': total / count, 'perplexity': math.exp(total / count), 'tokens': count}


def probe(model, x, y):
    """Separate diagnostic backward, without optimizer or data-order changes."""
    model.train(); model.zero_grad(set_to_none=True)
    activations, handles = {}, []
    for name, module in model.named_modules():
        if name.endswith('.ffn.up') or name.endswith('.ffn.down'):
            def capture(_module, _inputs, output, name=name):
                activations[name] = float(output.detach().float().square().mean().sqrt())
            handles.append(module.register_forward_hook(capture))
    try:
        _, loss = model(x, y); loss.backward()
        gradients = {}
        for name, p in model.named_parameters():
            if '.ffn.' in name and p.grad is not None:
                if not bool(torch.isfinite(p.grad).all()):
                    raise FloatingPointError('nonfinite FFN gradient: ' + name)
                gradients[name] = {'rms': float(p.grad.float().square().mean().sqrt()),
                                   'max_abs': float(p.grad.abs().max())}
        return {'loss': float(loss.detach()), 'activation_rms': activations, 'gradients': gradients}
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)


@torch.no_grad()
def generate(model):
    import tiktoken
    encoding = tiktoken.get_encoding('gpt2')
    results = []
    model.eval()
    for i, prompt in enumerate(('Once upon a time, there was a little', 'Lily found a red ball in the garden.')):
        ids = encoding.encode_ordinary(prompt)
        rng = torch.Generator(device='cuda').manual_seed(7000 + i)
        for _ in range(64):
            x = torch.tensor([ids[-model.config.context:]], device='cuda')
            logits = model(x)[0, -1] / 0.8
            threshold = torch.topk(logits, 40).values[-1]
            logits = logits.masked_fill(logits < threshold, -float('inf'))
            token = int(torch.multinomial(logits.softmax(-1), 1, generator=rng))
            ids.append(token)
            if token == encoding.eot_token:
                break
        results.append({'prompt': prompt, 'text': encoding.decode(ids), 'ids': ids,
                        'sampling_seed': 7000 + i, 'temperature': 0.8, 'top_k': 40})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--qualification-sha256', required=True)
    parser.add_argument('--data-manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kind', choices=('dense', 'cst', 'none'), required=True)
    parser.add_argument('--width', type=int, choices=(128, 256, 512), default=256)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--tokens', type=int, default=1000000)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--max-wall-seconds', type=int, default=3600)
    parser.add_argument('--repeat-fixed-batches', action='store_true')
    args = parser.parse_args(argv)
    if not 1024 <= args.tokens <= 20000000 or args.batch < 1 or args.accumulation < 1 or args.max_wall_seconds < 1:
        parser.error('bounded positive training budget required')
    if args.repeat_fixed_batches and args.tokens > 102400:
        parser.error('fixed-batch diagnostic is capped at100 standard updates')
    if sha(args.qualification) != args.qualification_sha256 or sha(args.data / 'manifest.json') != args.data_manifest_sha256:
        parser.error('qualification or data manifest identity differs from the frozen request')
    qualification = json.loads(args.qualification.read_text())
    if qualification['state'] != 'passed' or qualification['cohort'] != 'target':
        parser.error('passed target rectangular/multi-token qualification required')
    for name in ('ffn_model.py', 'ffn_contract.py'):
        if sha(Path(__file__).parent / name) != qualification['source_sha256'][name]:
            parser.error('model source differs from qualified source')
    pairs = {(x['in'], x['out']) for x in qualification['cases']}
    if (args.width, 4 * args.width) not in pairs or (4 * args.width, args.width) not in pairs:
        parser.error('qualification does not cover requested FFN dimensions')
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.get_device_name(0) == 'NVIDIA GeForce RTX 4090'
    manifest, data = verify_data(args.data)
    config = ModelConfig(width=args.width, vocabulary=manifest['vocabulary'])
    if args.batch * config.context > 256:
        parser.error('initial qualified microbatch has at most256 token rows')
    if args.tokens + 1 > len(data['train']):
        parser.error('train prefix does not cover token budget')
    model = LanguageModel(config, args.kind, args.seed).cuda()
    if args.kind == 'cst':
        configure_native(model)
        optimizer = cst_optimizer(model, lr=0.0003, betas=(0.9, 0.95), weight_decay=0.0, foreach=False)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, betas=(0.9, 0.95), weight_decay=0.0, foreach=False)
    tokens_per_update = args.batch * config.context * args.accumulation
    updates = args.tokens // tokens_per_update
    windows = args.tokens // config.context
    order = np.random.default_rng(args.seed + 888).permutation(windows)
    total_microbatches = updates * args.accumulation
    assert total_microbatches * args.batch <= len(order)
    px, py = token_batch(data['train'], [0], config.context, 'cuda')
    summary = {'state': 'running', 'config': asdict(config), 'arguments': {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
               'qualification_sha256': sha(args.qualification), 'data_manifest_sha256': sha(args.data / 'manifest.json'),
               'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
               'initialization_reports': model.initialization_reports(), 'ffn_budget': config.ffn_parameter_budget(),
               'total_parameters': sum(p.numel() for p in model.parameters()), 'scheduled_updates': updates,
               'scheduled_tokens': updates * tokens_per_update, 'completed_updates': 0, 'completed_tokens': 0,
               'optimizer': 'AdamW lr3e-4 betas(.9,.95) eps1e-8 decay0, global clip1; CST coordinate policy for managed atoms',
               'measurement': 'synchronized full optimizer update including accumulation/data transfer; no probes/evaluation in timed samples; eager FP32 TF32off',
               'diagnostics': {}, 'timing_samples_seconds': []}
    summary['peak_allocated_bytes'] = 0
    summary['peak_reserved_bytes'] = 0
    curves = (args.output / 'curves.jsonl').open('x')

    def save():
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    def log(entry):
        curves.write(json.dumps(entry) + '\n'); curves.flush()
        print(json.dumps(entry), flush=True)

    def checkpoint():
        target = args.output / 'checkpoint.pt'
        temporary = args.output / 'checkpoint.tmp'
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'config': asdict(config), 'kind': args.kind, 'seed': args.seed,
                    'updates': summary['completed_updates'], 'data_manifest_sha256': summary['data_manifest_sha256'],
                    'torch_rng_state': torch.get_rng_state(), 'cuda_rng_state': torch.cuda.get_rng_state()}, temporary)
        temporary.replace(target)
        summary['checkpoint_sha256'] = sha(target)

    try:
        summary['diagnostics']['initial'] = probe(model, px, py)
        initial = evaluate(model, data['development'], args.batch)
        log({'step': 0, 'seen_tokens': 0, 'elapsed_seconds': time.monotonic() - began, 'development': initial})
        summary['initial_development'] = initial
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        setup_end = time.monotonic(); summary['setup_seconds'] = setup_end - began
        warmup = max(1, min(200, updates // 10))
        for step in range(1, updates + 1):
            if time.monotonic() - began > args.max_wall_seconds:
                raise TimeoutError('bounded run wall budget reached; partial state preserved')
            model.train(); optimizer.zero_grad(set_to_none=True)
            if step <= warmup:
                lr = 0.0003 * step / warmup
            else:
                progress = (step - warmup) / max(1, updates - warmup)
                lr = 0.00003 + 0.00027 * (1 + math.cos(math.pi * progress)) / 2
            for group in optimizer.param_groups:
                group['lr'] = lr
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); tick = time.monotonic()
            loss_sum = torch.zeros((), device='cuda')
            for micro in range(args.accumulation):
                index = (micro if args.repeat_fixed_batches else ((step - 1) * args.accumulation + micro)) * args.batch
                starts = order[index:index + args.batch] * config.context
                x, y = token_batch(data['train'], starts, config.context, 'cuda')
                _, loss = model(x, y)
                (loss / args.accumulation).backward(); loss_sum += loss.detach()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step(); torch.cuda.synchronize()
            duration = time.monotonic() - tick
            summary['peak_allocated_bytes'] = max(summary['peak_allocated_bytes'], torch.cuda.max_memory_allocated())
            summary['peak_reserved_bytes'] = max(summary['peak_reserved_bytes'], torch.cuda.max_memory_reserved())
            value = float(loss_sum / args.accumulation)
            if not math.isfinite(value):
                raise FloatingPointError('nonfinite training loss')
            summary['completed_updates'] = step; summary['completed_tokens'] = step * tokens_per_update
            if step > 10:
                summary['timing_samples_seconds'].append(duration)
            if step <= 5 or step % 20 == 0 or step == updates:
                log({'step': step, 'seen_tokens': step * tokens_per_update, 'elapsed_seconds': time.monotonic() - began,
                     'train_ce': value, 'lr': lr, 'gradient_norm_before_clip': float(norm), 'complete_update_seconds': duration})
            if step in (10, 100):
                summary['diagnostics'][str(step)] = probe(model, px, py)
                torch.cuda.empty_cache()
            if step % 200 == 0 or step == updates:
                dev = evaluate(model, data['development'], args.batch)
                log({'step': step, 'seen_tokens': step * tokens_per_update, 'elapsed_seconds': time.monotonic() - began, 'development': dev})
                summary['latest_development'] = dev; checkpoint(); save()
        summary['diagnostics']['final'] = probe(model, px, py)
        if args.repeat_fixed_batches:
            summary['scope'] = 'repeated fixed-batch diagnostic; confirmation and generations not evaluated'
        else:
            summary['confirmation'] = evaluate(model, data['confirmation'], args.batch)
            summary['generations'] = generate(model)
        summary['state'] = 'completed_execution_not_scientific_admission'; checkpoint(); save()
    except Exception as error:
        summary['state'] = 'failed'; summary['error'] = f'{type(error).__name__}: {error}'
        checkpoint(); save()
        raise
    finally:
        summary['total_elapsed_seconds'] = time.monotonic() - began
        save(); curves.close()


if __name__ == '__main__':
    main()
