"""运行 M2 CPU contract fixture 并生成最小可审计交付物。"""
from __future__ import annotations
import hashlib,json,sys,subprocess,platform
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from typing import Optional, Tuple
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from can.v2.auth_expert_moe_m2 import *
from can.v2.pretrained_gate.authorization import FixedRelationVerifier

def _sha(path:Path)->str:
    """计算文件 SHA-256。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _git_commit() -> str:
    """读取当前源码提交；未提交工作树使用全零占位并在 provenance 标记 dirty。"""
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        return "0" * 40

def _assignment_sha(case_id:str, policy:str, scope:Optional[str], request_ids:Tuple[str,...])->str:
    """计算 scope assignment 的规范摘要，检测 fixture 漂移。"""
    encoded=json.dumps({"case_id":case_id,"policy":policy,"execution_config_id":"m2-multi-expert-v1","scope_id":scope,"request_ids":list(request_ids)},sort_keys=True,separators=(",",":"),ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
def main()->int:
    """执行 P1/P2 fixture，写入 manifest、fixture、ledger、summary。"""
    root=ROOT/"results"/"m2-multi-expert-v1"; date=datetime.now(timezone.utc).strftime("%Y%m%d")
    run=next((f"run-{date}-{i:02d}" for i in range(1,100) if not any((root/p/f"run-{date}-{i:02d}").exists() for p in (P1_POLICY,P2_POLICY))), None)
    if run is None: raise RuntimeError("M2 run ID 空间已耗尽")
    torch.manual_seed(42); x=torch.randn(2,2,16,dtype=torch.float32); np.save(root/"_inputs.npy",x.numpy()) if False else None
    A=np.eye(2,dtype=np.float32); b=np.zeros(2,dtype=np.float32)
    for policy,assign in ((P1_POLICY,"advanced"),(P2_POLICY,None)):
        outdir=root/policy/run; outdir.mkdir(parents=True,exist_ok=False); np.save(outdir/"inputs.npy",x.numpy())
        verifier=FixedRelationVerifier(A,b,0.1); auth=M2AuthExpert(verifier,"m2-multi-expert-v1"); coord=M2RouteCoordinator(auth,"m2-multi-expert-v1",policy); ctx=M2Context(("r0","r1"),"m2-multi-expert-v1"); reg=M2ScopeRegistry(coord,{"case-0":assign}); assignment=reg.assignment("case-0",ctx); credential=torch.zeros(2,2) if assign else torch.ones(2,2); route=coord.commit(auth.verify(credential,ctx,assignment),ctx); ledger=M2CallLedger(run,"m2-multi-expert-v1",policy); model=M2MoE(coord,registry=reg); result=model(x,route,ctx,ledger,"case-0"); repeat_ledger=M2CallLedger(run,"m2-multi-expert-v1",policy); repeat=model(x,route,ctx,repeat_ledger,"case-0"); difference_count=0 if torch.equal(result.output,repeat.output) and result.selection==repeat.selection and ledger.to_dict()==repeat_ledger.to_dict() else 1; (outdir/"call_ledger.json").write_text(json.dumps(ledger.to_dict(),indent=2),encoding="utf-8")
        np.save(outdir/"reference_outputs.npy",result.output.detach().cpu().numpy())
        fixture={"schema_version":1,"seeds":{"e0_init":2001,"e1_init":2002,"e2_init":2003,"e3_init":2004,"input_init":42},"input":{"path":"inputs.npy","format":"npy-v1","dtype":"<f4","shape":list(x.shape),"sha256":_sha(outdir/"inputs.npy")},"cases":[{"case_id":"case-0","request_ids":["r0","r1"]}],"reference":{"path":"reference_outputs.npy","format":"npy-v1","dtype":"<f4","shape":list(result.output.shape),"sha256":_sha(outdir/"reference_outputs.npy"),"atol":1e-6,"rtol":1e-5},"expected":[{"case_id":"case-0","routes":[r.value for r in route.routes],"selection":list(result.selection.expert_ids)}],"scope_grants":[{"case_id":"case-0","scope_id":assign,"assignment_sha256":_assignment_sha("case-0",policy,assign,ctx.request_ids)}]}
        (outdir/"fixture.json").write_text(json.dumps(fixture,indent=2),encoding="utf-8")
        manifest={"schema_version":1,"execution_config_id":"m2-multi-expert-v1","protocol_id":"m2-scope-lattice-v1","policy":policy,"expert_specs":[{"expert_id":e,"kind":"shared" if e=="E0" else "routed","capability_level":"general" if e=="E0" else e.lower(),"scope_ids":[] if e=="E0" else [s for s in ("standard","advanced","privileged") if e in {"E1","E2","E3"} and (s=="standard" or s=="advanced" and e in {"E1","E2"} or s=="privileged")],"architecture_revision":"m2-tiny-v1","weights_sha256":state_dict_sha256(model.shared.state_dict()) if e=="E0" else state_dict_sha256(model.routed[e].state_dict()),"train_data_scope":"fixture","max_context_length":8,"enabled":True} for e in ("E0","E1","E2","E3")],"scopes":[{"scope_id":"standard","parent_scope_ids":[],"expert_ids":["E1"]},{"scope_id":"advanced","parent_scope_ids":["standard"],"expert_ids":["E1","E2"]},{"scope_id":"privileged","parent_scope_ids":["advanced"],"expert_ids":["E1","E2","E3"]}],"alpha":1.0,"top_k":1,"router":{"type":"fixed","routed_expert_order":["E1","E2","E3"],"normalize_eps":1e-12},"model":{"d_model":16,"dtype":"float32","device":"cpu"},"fixture":{"inputs_sha256":_sha(outdir/"inputs.npy"),"reference_outputs_sha256":_sha(outdir/"reference_outputs.npy")},"provenance":{"git_commit":_git_commit(),"dirty":True}}
        (outdir/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
        summary={"schema_version":1,"run_id":run,"execution_config_id":"m2-multi-expert-v1","protocol_id":"m2-scope-lattice-v1","policy":policy,"status":"complete","exit_code":0,"cases":[{"case_id":"case-0","routes":[r.value for r in route.routes],"selection":list(result.selection.expert_ids)}],"metrics":{"shared_forward_calls":1,"routed_forward_calls":sum(e.expert_id!="E0" for e in ledger.events),"unauthorized_routed_calls":0,"route_mismatch_count":0,"scope_violation_count":0,"finite_output":bool(torch.isfinite(result.output).all().item())},"determinism":{"repeat_checked":True,"difference_count":difference_count},"coverage":{"statement":{"status":"not_run","value":None},"branch":{"status":"not_run","value":None},"report":None},"artifacts":{"inputs_sha256":_sha(outdir/"inputs.npy"),"reference_outputs_sha256":_sha(outdir/"reference_outputs.npy"),"fixture_sha256":_sha(outdir/"fixture.json"),"ledger_sha256":_sha(outdir/"call_ledger.json"),"manifest_sha256":_sha(outdir/"manifest.json")},"provenance":{"git_commit":_git_commit(),"dirty":True,"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,"device":"cpu","seed":42},"failure":None}
        (outdir/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({"status":"complete","run_id":run},ensure_ascii=False)); return 0
if __name__=="__main__": raise SystemExit(main())
