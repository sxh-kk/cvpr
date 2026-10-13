"""Fresh-process structural proof tests: no model decoding or tensor pages."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/probe_smplh_availability.py"
spec = importlib.util.spec_from_file_location("availability_under_test", SCRIPT)
availability = importlib.util.module_from_spec(spec); spec.loader.exec_module(availability)
sys.path.insert(0, str(SCRIPT.parent))
import cleanup_smplh_qualified as cleanup


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(availability.canonical(value))


@pytest.fixture
def structural(tmp_path):
    mix, hm = tmp_path / "Mix", tmp_path / "HM"
    root = mix / "parameters"; hm.mkdir()
    old = mix / "old.npy"; old.parent.mkdir(parents=True); old.write_bytes(b"blocked raw source")
    model = root / "weights/model.npz"; model.parent.mkdir(parents=True); model.write_bytes(b"not a model: SHA only, no decoding")
    specs = {key:{"dtype":"float32", "shape":[width]} for key,width in (
        ("smplh.root_orient",3),("smplh.pose_body",63),("smplh.trans",3),("smplh.betas",16))}
    specs.update({key:{"dtype":"bool","shape":[22]} for key in ("estimated.valid.body_position","estimated.valid.body_rotation")})
    put(root / "meta/info.json", {"model":{"path":"weights/model.npz","model_sha256":availability.sha(model)},
        "motion_features":specs,"optional_features":{}})
    records=[]
    for index in range(2):
        folder=root / "episodes" / str(index); folder.mkdir(parents=True)
        motion, vision=folder / "motion.parquet",folder / "vision.parquet"
        count=4
        fields=[]
        for name,s in specs.items():
            t=pa.type_for_alias(s["dtype"]); fields.append(pa.field(name,pa.list_(t,s["shape"][0])))
        schema=pa.schema(fields)
        pq.write_table(pa.Table.from_arrays([pa.array([[False if pa.types.is_boolean(f.type.value_type) else 0]*f.type.list_size]*count,type=f.type) for f in schema],schema=schema),motion)
        vs=pa.schema([pa.field("frame_index",pa.int64()),pa.field("episode_index",pa.int64()),pa.field("timestamp",pa.float64()),
                      pa.field("observation.valid.dino",pa.bool_()),pa.field("observation.features.dino",pa.list_(pa.float32(),5120)),
                      pa.field("observation.source_timestamp",pa.float64()),pa.field("source.video_frame_index",pa.int64()),
                      pa.field("observation.video_timestamp",pa.float64()),pa.field("observation.video_time_error",pa.float64())])
        pq.write_table(pa.Table.from_arrays([pa.array([0]*count,type=pa.int64()),pa.array([index]*count,type=pa.int64()),
            pa.array([0.0]*count,type=pa.float64()),pa.array([False]*count,type=pa.bool_()),pa.array([[0]*5120]*count,type=pa.list_(pa.float32(),5120)),
            pa.array([0.0]*count,type=pa.float64()),pa.array([-1]*count,type=pa.int64()),pa.array([0.0]*count,type=pa.float64()),
            pa.array([0.0]*count,type=pa.float64())],schema=vs),vision)
        record=dict(episode_index=index,source="egosuite",source_key=f"egosuite:{index}",length=count,split="train",
                    data_path=str(motion.relative_to(root)),vision_path=str(vision.relative_to(root)),
                    motion_sha256=availability.sha(motion),vision_sha256=availability.sha(vision))
        receipt=dict(record=record,output_identities={"motion":availability.identity(motion),"vision":availability.identity(vision)},
                     motion={"sha256":record["motion_sha256"]},vision={"sha256":record["vision_sha256"]})
        put(folder / "receipt.json",receipt);put(root / ".pipeline/episodes" / f"{index:08d}.json",record);records.append(record)
    protected={}
    for path in root.rglob('*'):
        if path.is_file():
            protected[str(path)]=dict(identity=availability.identity(path),sha256=availability.sha(path),binary=path.suffix=='.parquet')
    fields=("episode_index","source","source_key","length","split","data_path","vision_path")
    env=dict(root=str(root),blocked_roots=[str(mix),str(hm)],records=[{k:r[k] for k in fields} for r in records],
        protected_files=protected,manifest_sha256="a"*64,model_sha256=availability.sha(model),
        original_numeric_qualification_sha256="b"*64,population=dict(episodes=2,frames=8))
    return dict(root=root,mix=mix,hm=hm,old=old,records=records,envelope=env)


def run(envelope):
    result=subprocess.run([sys.executable,str(SCRIPT)],input=json.dumps(envelope),text=True,capture_output=True,timeout=30)
    assert result.stdout,result.stderr
    return result,json.loads(result.stdout)


def test_fresh_availability_only_checks_headers_even_with_non_model_asset(structural):
    result,report=run(structural['envelope'])
    assert result.returncode==0 and report['passed'],report
    assert report['scope']=='structural_availability_only' and report['fresh_process']
    assert report['episodes']==2 and report['frames']==8
    assert report['numerical_checks_rerun'] is False and report['fk_or_model_executed'] is False and report['data_pages_read'] is False
    assert report['old_source_open_attempts']==[] and report['pipeline_imports']==[]
    assert all(row['motion_footer_rows']==row['vision_footer_rows']==4 for row in report['rows'])
    # The asset is deliberately not a valid NPZ. Passing proves no decoder/model
    # loader is used by this structural stage.


@pytest.mark.parametrize('kind',['missing_population','changed_output','symlink','wrong_schema','wrong_vision_schema'])
def test_availability_rejects_missing_changed_and_non_owned_dependencies(structural,kind):
    env=deepcopy(structural['envelope'])
    record=structural['records'][0];motion=structural['root']/record['data_path']
    if kind=='missing_population':env['records'].pop()
    elif kind=='changed_output':motion.write_bytes(b'changed output')
    elif kind=='symlink':motion.unlink();motion.symlink_to(structural['old'])
    else:
        # Rebind identity/receipt for this isolated fixture to prove the footer
        # schema itself rejects a missing parameter column.
        role='vision' if kind=='wrong_vision_schema' else 'motion'
        target=structural['root']/record[role=='vision' and 'vision_path' or 'data_path']
        pq.write_table(pa.table({'unrelated':[0,0,0,0]}),target)
        receipt_path=motion.parent/'receipt.json';receipt=json.loads(receipt_path.read_text())
        receipt['output_identities'][role]=availability.identity(target);receipt['record'][role+'_sha256']=availability.sha(target)
        receipt[role]['sha256']=availability.sha(target);put(receipt_path,receipt)
        path=structural['root']/'.pipeline/episodes/00000000.json';put(path,receipt['record'])
        for p in (path,receipt_path,target):env['protected_files'][str(p)].update(identity=availability.identity(p),sha256=availability.sha(p))
    result,report=run(env)
    assert result.returncode!=0 and not report['passed']
    assert report['numerical_checks_rerun'] is False and report['fk_or_model_executed'] is False


def test_guard_in_fresh_process_rejects_raw_and_numeric_reader_imports(structural):
    code=r'''
import importlib.util,json,sys
from pathlib import Path
p=json.loads(sys.argv[1]);s=importlib.util.spec_from_file_location("availability",p["script"]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
g=m.Guard(p["root"],[p["mix"],p["hm"]]).install()
for action in (lambda:Path(p["old"]).read_bytes(),lambda:__import__("egomotion_pipeline.body_model"),lambda:__import__("MotionExpert.dataset.smplh_archive"),lambda:(g.motionexpert_root/"dataset/smplh_archive.py").read_bytes()):
 try:action()
 except (PermissionError,ImportError):continue
 raise AssertionError("guard allowed numeric/source dependency")
assert len(g.old_source_open_attempts)==1 and len(g.pipeline_imports)==3
print("passed")
'''
    value={k:str(structural[k]) for k in ('root','mix','hm','old')};value['script']=str(SCRIPT)
    result=subprocess.run([sys.executable,'-c',code,json.dumps(value)],text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stderr


def test_probe_never_uses_data_page_read_methods(structural):
    code=r'''
import importlib.util,json,sys
import pyarrow.parquet as pq
s=importlib.util.spec_from_file_location("availability",sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
original=pq.ParquetFile
class FooterOnly:
 def __init__(self,*args,**kwargs):self.file=original(*args,**kwargs)
 def __getattr__(self,name):
  if name.startswith("read") or name=="iter_batches":raise AssertionError("numeric/data-page method used")
  return getattr(self.file,name)
def forbidden(*args,**kwargs):raise AssertionError("table data-page read used")
pq.ParquetFile=FooterOnly;pq.read_table=forbidden;pq.read_pandas=forbidden
r=m.run(json.load(sys.stdin));assert r["passed"],r;assert r["data_pages_read"] is False
print(json.dumps(r))
'''
    result=subprocess.run([sys.executable,'-c',code,str(SCRIPT)],input=json.dumps(structural['envelope']),text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['passed']


def test_cleanup_post_stage_runs_only_independent_availability_process(structural,tmp_path):
    _,report=run(structural['envelope'])
    ctx=dict(root=structural['root'],mix=structural['mix'],job=tmp_path/'job',records=structural['records'],population=report['population'],
        plan=dict(egosuite_fitted_root=str(structural['mix']/'fits'),hm_root=str(structural['hm'])),
        report=dict(binding=dict(model_sha256=report['model_sha256']),proofs=dict(original_fit_qualification=dict(sha256='b'*64))))
    manifest=dict(manifest_sha256='a'*64,protected_files=structural['envelope']['protected_files'])
    result=cleanup.post_availability_probe(ctx,manifest)
    assert result['scope']=='structural_availability_only' and result['passed']
    saved=json.loads(Path(result['path']).read_text())
    assert saved['numerical_checks_rerun'] is False and saved['fk_or_model_executed'] is False
    assert saved['episodes']==2


@pytest.mark.parametrize('kind',['numeric_claim','partial','old_access','legacy_format'])
def test_new_verifier_cannot_accept_numeric_scope_partial_population_or_old_access(structural,kind):
    _,report=run(structural['envelope'])
    ctx=dict(root=structural['root'],mix=structural['mix'],records=structural['records'],population=report['population'],
        plan=dict(egosuite_fitted_root=str(structural['mix']/'fits'),hm_root=str(structural['hm'])),
        report=dict(binding=dict(model_sha256=report['model_sha256']),proofs=dict(original_fit_qualification=dict(sha256='b'*64))))
    manifest=dict(manifest_sha256='a'*64,protected_files=structural['envelope']['protected_files'])
    report['blocked_roots'].append(ctx['plan']['egosuite_fitted_root'])
    cleanup.verify_availability(report,ctx,manifest)
    if kind=='numeric_claim':report['fk_or_model_executed']=True
    elif kind=='partial':report['rows'].pop()
    elif kind=='old_access':report['old_source_open_attempts']=[{'path':str(structural['old'])}]
    else:report['format']='independent-standalone-smplh-parameter-fresh-reader-v1'
    with pytest.raises(ValueError,match='structural-only contract'):cleanup.verify_availability(report,ctx,manifest)
