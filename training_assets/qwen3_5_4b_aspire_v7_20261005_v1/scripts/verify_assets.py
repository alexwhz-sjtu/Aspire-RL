"""Offline asset verification and configuration parsing; never launch a trainer."""
import argparse
import json
import os
import pathlib
import subprocess
import sys

parser=argparse.ArgumentParser()
parser.add_argument('--asset-root',type=pathlib.Path,required=True)
parser.add_argument('--full-model',action='store_true')
args=parser.parse_args()
asset=args.asset_root.resolve()
root=pathlib.Path('/data/wanghanzhen/projects/EmbodiedAReal')
sys.path[:0]=[str(root/'AReaL'),str(root/'ASPIRE')]
from aspire.sim.online_rl.snapshots import sha256_file,verify_snapshot,processor_digest
from aspire.sim.online_rl.task_registry import TaskRegistry
meta=json.loads((asset/'provenance/asset_manifest.json').read_text())
registry=TaskRegistry(asset/'datasets/train_campaign.json')
assert len(registry.manifest.tasks)==10
verify_snapshot(asset/'snapshots/skill-v7',meta['skill_digest'])
model=json.loads((asset/'models/base_model.json').read_text())
assert processor_digest(model['path'])==model['processor_digest']
for name,entry in model['files'].items():
    path=pathlib.Path(model['path'])/name
    assert path.stat().st_size==entry['size_bytes'],name
    if args.full_model or not name.endswith('.safetensors'):
        assert sha256_file(path)==entry['sha256'],name
hashes=json.loads((asset/'provenance/source_hashes.json').read_text())
for relative,expected in hashes.items():
    assert sha256_file(root/relative)==expected,f'working source changed: {relative}'
    assert sha256_file(asset/'provenance/source_snapshot'/relative)==expected,f'archived source changed: {relative}'
for repo,key in [('AReaL','areal_commit'),('ASPIRE','aspire_commit')]:
    assert subprocess.check_output(['git','-C',str(root/repo),'rev-parse','HEAD'],text=True).strip()==meta[key]
for path in (asset/'snapshots').rglob('*'):
    assert not path.is_symlink(),f'snapshot symlink: {path}'
os.environ.setdefault('ASPIRE_BROKER_URL','http://127.0.0.1:8625')
os.environ.setdefault('ASPIRE_ACTOR_MODEL_PATH',model['path'])
os.environ.setdefault('ASPIRE_CAMPAIGN_MANIFEST',str(asset/'datasets/train_campaign.json'))
os.environ.setdefault('AREAL_ADMIN_API_KEY','static-audit-only')
os.environ.setdefault('AREAL_EXPERIMENT_ROOT','/tmp/aspire-v7-audit-not-launched')
os.environ.setdefault('AREAL_NAME_RESOLVE_ROOT','/tmp/aspire-v7-audit-not-launched/name-resolve')
from examples.aspire.config import AspireGRPOConfig
from areal.api.cli_args import load_expr_config
from examples.aspire.dataset import load_campaign_rows
config,_=load_expr_config(['--config',str(asset/'configs/acceptance_lora.yaml')],AspireGRPOConfig)
rows=load_campaign_rows(asset/'datasets/train_campaign.json')
assert len(rows)==10 and set(rows.column_names)=={'task_uid'}
assert config.actor.backend=='fsdp:d2p1t1' and config.rollout.backend=='sglang:d1p1t4'
assert config.gconfig.n_samples==4 and config.train_dataset.batch_size==2
print(json.dumps({'status':'passed_offline_only','tasks':len(rows),'skill_digest':meta['skill_digest'],'source_files':len(hashes),'full_model_hash_verified':args.full_model,'config_parsed':True,'services_started':False,'training_started':False,'gpu_canaries':'pending'},indent=2))
