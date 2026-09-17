"""M2 多 expert/scope contract 专项测试。"""
import numpy as np
import pytest
import torch
import hashlib
import json
from pathlib import Path
from can.v2.pretrained_gate.authorization import FixedRelationVerifier
from can.v2.auth_expert_moe_m2 import *
from can.v2.pretrained_gate.types import RouteKind
from can.v2.auth_expert_moe_m2.artifacts import load_m2_fixture, validate_m2_call_ledger, validate_m2_summary
from can.v2.auth_expert_moe_m2.manifest import load_m2_manifest
from can.v2.auth_expert_moe_m2 import authentication as m2_authentication

def _setup(policy=P1_POLICY, assignment="advanced"):
    """创建固定 A0、协调器、registry 和上下文。"""
    v=FixedRelationVerifier(np.eye(2,dtype=np.float32),np.zeros(2,dtype=np.float32),0.1)
    auth=M2AuthExpert(v,"m2-multi-expert-v1"); coord=M2RouteCoordinator(auth,"m2-multi-expert-v1",policy)
    ctx=M2Context(("r0","r1"),"m2-multi-expert-v1"); reg=M2ScopeRegistry(coord,{"case":assignment}); return auth,coord,ctx,reg

def test_fixed_relation_verifier_satisfies_public_protocol():
    """确认原 A0 无需包装即可满足公开 verifier protocol。"""
    verifier=FixedRelationVerifier(np.eye(2,dtype=np.float32),np.zeros(2,dtype=np.float32),0.1)
    assert isinstance(verifier,M2VerifierProtocol)

def test_lattice_and_router_determinism():
    assert validate_scope_lattice([{"scope_id":"standard","parent_scope_ids":[],"expert_ids":["E1"]},{"scope_id":"advanced","parent_scope_ids":["standard"],"expert_ids":["E1","E2"]},{"scope_id":"privileged","parent_scope_ids":["advanced"],"expert_ids":["E1","E2","E3"]}])["privileged"]==("E1","E2","E3")
    r=FixedConstrainedRouter(); out=r.select(torch.zeros(2,3),torch.tensor([[1,1,0],[0,0,0]],dtype=torch.bool)); assert out.expert_ids==("E1",None)

def test_scope_rejects_bad_parent():
    with pytest.raises(Exception): validate_scope_lattice([{"scope_id":"standard","parent_scope_ids":["x"],"expert_ids":["E1"]},{"scope_id":"advanced","parent_scope_ids":["standard"],"expert_ids":["E1","E2"]},{"scope_id":"privileged","parent_scope_ids":["advanced"],"expert_ids":["E1","E2","E3"]}])

def test_moe_shared_and_routed_calls():
    auth,coord,ctx,reg=_setup(); ass=reg.assignment("case",ctx); route=coord.commit(auth.verify(torch.zeros(2,2),ctx,ass),ctx); ledger=M2CallLedger("run","m2-multi-expert-v1",P1_POLICY); out=M2MoE(coord,registry=reg)(torch.ones(2,2,16),route,ctx,ledger,"case"); assert out.selection.expert_ids==("E1","E1"); assert [e.expert_id for e in ledger.events]==["E0","E1"]

def test_router_can_select_e2_and_e3_with_mask():
    r=FixedConstrainedRouter(); out=r.select(torch.tensor([[0.,5.,9.],[0.,-2.,3.]]),torch.tensor([[0,1,1],[0,0,1]],dtype=torch.bool)); assert out.expert_ids==("E3","E3")

def test_denied_zero_routed_calls():
    auth,coord,ctx,reg=_setup(assignment=None); ass=reg.assignment("case",ctx); route=coord.commit(auth.verify(torch.ones(2,2),ctx,ass),ctx); ledger=M2CallLedger("run","m2-multi-expert-v1",P1_POLICY); out=M2MoE(coord,registry=reg)(torch.ones(2,1,16),route,ctx,ledger,"case"); assert out.selection.expert_ids==(None,None); assert [e.expert_id for e in ledger.events]==["E0"]

def test_p2_relation_failure_public_shared_only():
    auth,coord,ctx,reg=_setup(P2_POLICY,"standard"); ass=reg.assignment("case",ctx); route=coord.commit(auth.verify(torch.ones(2,2),ctx,ass),ctx); assert route.routes==(RouteKind.PUBLIC,RouteKind.PUBLIC)

def test_route_binding_rejects_foreign_context():
    auth,coord,ctx,reg=_setup(); ass=reg.assignment("case",ctx); route=coord.commit(auth.verify(torch.zeros(2,2),ctx,ass),ctx)
    with pytest.raises(Exception): coord.validate(route,M2Context(("x","y"),"m2-multi-expert-v1"))

def test_router_rejects_bad_shapes_and_dtype():
    r=FixedConstrainedRouter()
    with pytest.raises(ValueError): r.select(torch.zeros(1,2),torch.zeros(1,2,dtype=torch.bool))
    with pytest.raises(ValueError): r.select(torch.zeros(1,3,dtype=torch.int64),torch.zeros(1,3,dtype=torch.bool))
    with pytest.raises(ValueError): r.select(torch.zeros(1,3),torch.zeros(1,3,dtype=torch.int64))

def test_scope_rejects_schema_order_duplicates_and_expert():
    base=[{"scope_id":"standard","parent_scope_ids":[],"expert_ids":["E1"]},{"scope_id":"advanced","parent_scope_ids":["standard"],"expert_ids":["E1","E2"]},{"scope_id":"privileged","parent_scope_ids":["advanced"],"expert_ids":["E1","E2","E3"]}]
    with pytest.raises(ValueError): validate_scope_lattice(base[:2])
    dup=[dict(x) for x in base]; dup[1]["scope_id"]="standard"
    with pytest.raises(ValueError): validate_scope_lattice(dup)
    bad=[dict(x) for x in base]; bad[2]["expert_ids"]=["E1","E2"]
    with pytest.raises(Exception): validate_scope_lattice(bad)
    cyc=[dict(x) for x in base]; cyc[0]["parent_scope_ids"]=["privileged"]
    with pytest.raises(Exception): validate_scope_lattice(cyc)

def test_scope_registry_rejects_unknown_and_expansion():
    auth,coord,ctx,reg=_setup()
    with pytest.raises(Exception): reg.assignment("missing",ctx)
    with pytest.raises(Exception): M2ScopeRegistry(coord,{"case":"unknown"}).assignment("case",ctx)
    ass=reg.assignment("case",ctx); route=coord.commit(auth.verify(torch.zeros(2,2),ctx,ass),ctx)
    view=reg.resolve(route,ctx)
    with pytest.raises(Exception): reg.restrict(view,(("E1","E2","E3"),("E1","E2")))
    with pytest.raises(Exception): reg.restrict(view,(("E1",),))

def test_m2_artifact_and_manifest_loaders(tmp_path: Path):
    x=np.zeros((2,2,16),dtype="<f4"); np.save(tmp_path/"inputs.npy",x); np.save(tmp_path/"reference_outputs.npy",x)
    sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    fixture={"schema_version":1,"seeds":{"e0_init":1,"e1_init":2,"e2_init":3,"e3_init":4,"input_init":5},"input":{"path":"inputs.npy","format":"npy-v1","dtype":"<f4","shape":[2,2,16],"sha256":sha(tmp_path/"inputs.npy")},"cases":[{"case_id":"case-0"}],"reference":{"path":"reference_outputs.npy","format":"npy-v1","dtype":"<f4","shape":[2,2,16],"sha256":sha(tmp_path/"reference_outputs.npy")},"expected":[],"scope_grants":[{"case_id":"case-0","scope_id":None,"assignment_sha256":"0"*64}]}
    (tmp_path/"fixture.json").write_text(json.dumps(fixture),encoding="utf-8"); assert load_m2_fixture(tmp_path/"fixture.json")["schema_version"]==1
    ledger={"schema_version":1,"run_id":"r","execution_config_id":"m2-multi-expert-v1","policy":P1_POLICY,"events":[]}; validate_m2_call_ledger(ledger)
    summary={"schema_version":1,"run_id":"r","execution_config_id":"m2-multi-expert-v1","protocol_id":"m2-scope-lattice-v1","policy":P1_POLICY,"status":"complete","exit_code":0,"cases":[],"metrics":{},"determinism":{},"coverage":{},"artifacts":{},"provenance":{},"failure":None}; validate_m2_summary(summary)

def test_m2_artifact_rejects_tamper_and_bad_schema(tmp_path: Path):
    (tmp_path/"fixture.json").write_text("{}",encoding="utf-8")
    with pytest.raises(ValueError): load_m2_fixture(tmp_path/"fixture.json")
    with pytest.raises(FileNotFoundError): load_m2_fixture(tmp_path/"missing.json")
    with pytest.raises(ValueError): validate_m2_call_ledger({"schema_version":1})
    with pytest.raises(ValueError): validate_m2_summary({"schema_version":1})

def test_manifest_valid_and_digest_rejected(tmp_path: Path):
    scopes=[{"scope_id":"standard","parent_scope_ids":[],"expert_ids":["E1"]},{"scope_id":"advanced","parent_scope_ids":["standard"],"expert_ids":["E1","E2"]},{"scope_id":"privileged","parent_scope_ids":["advanced"],"expert_ids":["E1","E2","E3"]}]
    specs=[{"expert_id":e,"kind":"shared" if e=="E0" else "routed","capability_level":"general","scope_ids":[],"architecture_revision":"m2-tiny-v1","weights_sha256":"0"*64,"train_data_scope":"fixture","max_context_length":8,"enabled":True} for e in ("E0","E1","E2","E3")]
    payload={"schema_version":1,"execution_config_id":"m2-multi-expert-v1","protocol_id":"m2-scope-lattice-v1","policy":P1_POLICY,"expert_specs":specs,"scopes":scopes,"alpha":1.0,"top_k":1,"router":{"type":"fixed","routed_expert_order":["E1","E2","E3"],"normalize_eps":1e-12},"model":{"d_model":16,"dtype":"float32","device":"cpu"},"fixture":{},"provenance":{"git_commit":"0"*40,"dirty":True}}
    path=tmp_path/"manifest.json"; path.write_text(json.dumps(payload),encoding="utf-8"); digest=hashlib.sha256(path.read_bytes()).hexdigest(); assert load_m2_manifest(path,digest)["schema_version"]==1
    with pytest.raises(ValueError): load_m2_manifest(path,"0"*64)
    payload["top_k"]=2; path.write_text(json.dumps(payload),encoding="utf-8")
    with pytest.raises(ValueError): load_m2_manifest(path,hashlib.sha256(path.read_bytes()).hexdigest())

def test_authentication_exports_are_consistent():
    assert m2_authentication.M2AuthExpert is M2AuthExpert

def test_ledger_and_summary_reject_malformed_events():
    valid={"schema_version":1,"run_id":"r","execution_config_id":"m2-multi-expert-v1","policy":P1_POLICY,"events":[{"sequence":0,"case_id":"c","stage":"forward","expert_id":"E0","kind":"shared","batch_indices":[0],"count":1}]}
    validate_m2_call_ledger(valid)
    for event in ({**valid["events"][0],"sequence":1},{**valid["events"][0],"count":2},{**valid["events"][0],"batch_indices":[]}):
        bad={**valid,"events":[event]}
        with pytest.raises(ValueError): validate_m2_call_ledger(bad)

def test_manifest_rejects_duplicate_and_nested_fields(tmp_path: Path):
    p=tmp_path/"bad.json"; p.write_text('{"schema_version":1,"schema_version":1}',encoding="utf-8")
    with pytest.raises(FileNotFoundError): load_m2_manifest("not-a-path","0"*64)
    with pytest.raises(ValueError): load_m2_manifest(p,hashlib.sha256(p.read_bytes()).hexdigest())
    p.write_text("{}",encoding="utf-8")
    with pytest.raises(ValueError): load_m2_manifest(p,hashlib.sha256(p.read_bytes()).hexdigest())

def test_route_validation_rejects_invalid_allowed_sets():
    auth,coord,ctx,reg=_setup(); ass=reg.assignment("case",ctx); route=coord.commit(auth.verify(torch.zeros(2,2),ctx,ass),ctx)
    with pytest.raises(Exception):
        coord.validate(M2CommittedRoute(route.routes,(("E0",),("E1",)),ctx,P1_POLICY,"case",coord._seal),ctx)
    with pytest.raises(Exception):
        coord.validate(M2CommittedRoute((RouteKind.DENY,RouteKind.DENY),(("E1",),()),ctx,P1_POLICY,"case",coord._seal),ctx)

def test_moe_public_and_foreign_view_rejected():
    auth,coord,ctx,reg=_setup(P2_POLICY,"advanced"); ass=reg.assignment("case",ctx); route=coord.commit(auth.verify(torch.ones(2,2),ctx,ass),ctx)
    with pytest.raises(Exception): M2MoE(coord,registry=reg)(torch.ones(2,1,16),route,ctx,view=M2AllowedExperts(route.allowed_experts,route,ctx,object()))

@pytest.mark.parametrize("scope,expected", [("standard",("E1",)), ("privileged",("E1","E2","E3"))])
def test_each_scope_produces_exact_allowed_set(scope, expected):
    auth,coord,ctx,reg=_setup(P1_POLICY,scope); ass=reg.assignment("case",ctx); route=coord.commit(auth.verify(torch.zeros(2,2),ctx,ass),ctx)
    assert route.allowed_experts==(expected,expected)

@pytest.mark.parametrize("batch", [1,4])
def test_batch_sizes_and_finite_outputs(batch):
    v=FixedRelationVerifier(np.eye(2,dtype=np.float32),np.zeros(2,dtype=np.float32),0.1); auth=M2AuthExpert(v,"m2-multi-expert-v1"); coord=M2RouteCoordinator(auth,"m2-multi-expert-v1"); ctx=M2Context(tuple(f"r{i}" for i in range(batch)),"m2-multi-expert-v1"); reg=M2ScopeRegistry(coord,{"case":"standard"}); ass=reg.assignment("case",ctx); route=coord.commit(auth.verify(torch.zeros(batch,2),ctx,ass),ctx); out=M2MoE(coord,registry=reg)(torch.ones(batch,1,16),route,ctx); assert out.output.shape==(batch,1,16) and bool(torch.isfinite(out.output).all())

def test_router_masks_high_unauthorized_logit():
    out=FixedConstrainedRouter().select(torch.tensor([[100.,0.,0.]]),torch.tensor([[False,True,False]])); assert out.expert_ids==("E2",)

def test_deterministic_repeated_runs():
    def run_once():
        torch.manual_seed(7); auth,coord,ctx,reg=_setup(P1_POLICY,"privileged"); ass=reg.assignment("case",ctx); route=coord.commit(auth.verify(torch.zeros(2,2),ctx,ass),ctx); ledger=M2CallLedger("run","m2-multi-expert-v1",P1_POLICY); out=M2MoE(coord,registry=reg)(torch.randn(2,3,16),route,ctx,ledger); return route.routes,out.selection.expert_ids,out.output.detach(),ledger.to_dict()
    a=run_once(); b=run_once(); assert a[0]==b[0] and a[1]==b[1] and torch.equal(a[2],b[2]) and a[3]==b[3]

def test_expert_exception_does_not_return_partial_output():
    class Broken(RoutedExpert):
        """用于验证异常时不返回部分输出的测试专家。"""
        def forward(self,*args,**kwargs):
            """注入确定性异常。"""
            raise RuntimeError("injected")
    auth,coord,ctx,reg=_setup(); ass=reg.assignment("case",ctx); route=coord.commit(auth.verify(torch.zeros(2,2),ctx,ass),ctx); experts={"E1":Broken("E1",2002)}
    with pytest.raises(RuntimeError): M2MoE(coord,experts=experts,registry=reg)(torch.ones(2,1,16),route,ctx)

def test_mixed_allowed_mask_keeps_original_rows():
    out=FixedConstrainedRouter().select(torch.zeros(3,3),torch.tensor([[True,False,False],[True,True,False],[True,True,True]])); assert out.expert_ids==("E1","E1","E1")

@pytest.mark.parametrize("kwargs", [
    {"execution_config_id":""}, {"protocol_id":""}, {"d_model":True},
    {"d_model":0}, {"top_k":2}, {"alpha":0.5}, {"normalize_eps":1.0},
    {"dtype":torch.int64},
])
def test_config_rejects_unfrozen_values(kwargs):
    with pytest.raises((ValueError,TypeError)): M2Config(**kwargs)

@pytest.mark.parametrize("ids", [(), ("",), ("r0","r0"), ["r0"]])
def test_context_rejects_bad_request_ids(ids):
    with pytest.raises((ValueError,TypeError)): M2Context(ids,"m2-multi-expert-v1")

def test_ledger_rejects_bad_indices_and_serializes():
    ledger=M2CallLedger("run","m2-multi-expert-v1",P1_POLICY)
    for indices in ((),(-1,), (True,), ("0",)):
        with pytest.raises(ValueError): ledger.record("c","E0","shared",indices)
    ledger.record("c","E0","shared",(0,)); assert ledger.to_dict()["events"][0]["count"]==1

@pytest.mark.parametrize("expert_id", ["E0","E4",None])
def test_routed_expert_rejects_unknown_id(expert_id):
    with pytest.raises(ValueError): RoutedExpert(expert_id,1)

def test_expert_rejects_shape_dtype_and_nonfinite():
    expert=SharedExpert()
    for hidden in (torch.zeros(16),torch.zeros(1,1,15),torch.zeros(1,1,16,dtype=torch.float64),torch.full((1,1,16),float("nan"))):
        with pytest.raises(ValueError): expert(hidden,("r",))
    with pytest.raises(ValueError): expert(torch.zeros(1,1,16),("r","s"))

def test_auth_rejects_unregistered_assignment():
    auth,coord,ctx,reg=_setup(); ass=M2ScopeAssignment("case","standard",ctx.request_ids,P1_POLICY,ctx.execution_config_id,object())
    with pytest.raises(M2AuthorizationError): auth.verify(torch.zeros(2,2),ctx,ass)

def test_auth_rejects_assignment_context_mismatch():
    auth,coord,ctx,reg=_setup(); ass=reg.assignment("case",ctx); foreign=M2Context(("x","y"),ctx.execution_config_id)
    with pytest.raises(M2AuthorizationError): auth.verify(torch.zeros(2,2),foreign,ass)

def test_coordinator_rejects_bad_policy_and_config():
    auth,_,_,_=_setup()
    with pytest.raises(ValueError): M2RouteCoordinator(auth,"other")
    with pytest.raises(ValueError): M2RouteCoordinator(auth,"m2-multi-expert-v1","bad")

def test_moe_rejects_high_logit_scope_bypass():
    class UnsafeRouter:
        """返回未授权 expert 的测试 Router。"""
        def select(self,logits,mask):
            """故意选择 E3。"""
            return M2Selection(("E3","E3"))
    auth,coord,ctx,reg=_setup(P1_POLICY,"standard"); ass=reg.assignment("case",ctx); route=coord.commit(auth.verify(torch.zeros(2,2),ctx,ass),ctx)
    with pytest.raises(M2AuthorizationError): M2MoE(coord,router=UnsafeRouter(),registry=reg)(torch.ones(2,1,16),route,ctx)

def test_moe_routes_e2_and_e3_for_privileged_rows():
    class FixedRouter:
        """测试用受限选择器，验证 E2/E3 的真实 dispatch。"""
        def select(self,logits,mask):
            """逐行返回 E2、E3。"""
            return M2Selection(("E2","E3"))
    auth,coord,ctx,reg=_setup(P1_POLICY,"privileged"); ass=reg.assignment("case",ctx); route=coord.commit(auth.verify(torch.zeros(2,2),ctx,ass),ctx); ledger=M2CallLedger("run","m2-multi-expert-v1",P1_POLICY); out=M2MoE(coord,router=FixedRouter(),registry=reg)(torch.ones(2,1,16),route,ctx,ledger)
    assert out.selection.expert_ids==("E2","E3") and [e.expert_id for e in ledger.events]==["E0","E2","E3"]

def test_p2_public_path_runs_shared_only():
    auth,coord,ctx,reg=_setup(P2_POLICY,None); ass=reg.assignment("case",ctx); route=coord.commit(auth.verify(torch.ones(2,2),ctx,ass),ctx); ledger=M2CallLedger("run","m2-multi-expert-v1",P2_POLICY); out=M2MoE(coord,registry=reg)(torch.ones(2,1,16),route,ctx,ledger); assert out.routed_indices.numel()==0 and [e.expert_id for e in ledger.events]==["E0"]

def test_scope_restrict_empty_is_valid_zero_call():
    auth,coord,ctx,reg=_setup(); ass=reg.assignment("case",ctx); route=coord.commit(auth.verify(torch.zeros(2,2),ctx,ass),ctx); view=reg.resolve(route,ctx); narrowed=reg.restrict(view,((),())); ledger=M2CallLedger("run","m2-multi-expert-v1",P1_POLICY); out=M2MoE(coord,registry=reg)(torch.ones(2,1,16),route,ctx,ledger,view=narrowed); assert out.routed_indices.numel()==0

def _fixture_payload(tmp_path: Path):
    """生成用于负向 schema 测试的基础 fixture。"""
    x=np.zeros((1,1,16),dtype="<f4"); np.save(tmp_path/"inputs.npy",x); np.save(tmp_path/"reference_outputs.npy",x); sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    return {"schema_version":1,"seeds":{"e0_init":1,"e1_init":2,"e2_init":3,"e3_init":4,"input_init":5},"input":{"path":"inputs.npy","format":"npy-v1","dtype":"<f4","shape":[1,1,16],"sha256":sha(tmp_path/"inputs.npy")},"cases":[{"case_id":"case-0"}],"reference":{"path":"reference_outputs.npy","format":"npy-v1","dtype":"<f4","shape":[1,1,16],"sha256":sha(tmp_path/"reference_outputs.npy")},"expected":[],"scope_grants":[{"case_id":"case-0","scope_id":None,"assignment_sha256":"0"*64}]}

@pytest.mark.parametrize("section,change", [("input",{"path":"bad.npy"}),("input",{"dtype":"<f8"}),("input",{"sha256":"0"*64}),("reference",{"shape":[9,9,9]})])
def test_fixture_rejects_each_artifact_mismatch(tmp_path: Path, section, change):
    payload=_fixture_payload(tmp_path); payload[section].update(change); p=tmp_path/"fixture.json"; p.write_text(json.dumps(payload),encoding="utf-8")
    with pytest.raises(ValueError): load_m2_fixture(p)

@pytest.mark.parametrize("change", [{"scope_grants":None},{"scope_grants":[{"case_id":"other","scope_id":None,"assignment_sha256":"0"*64}]},{"scope_grants":[{"case_id":"case-0","scope_id":"bad","assignment_sha256":"0"*64}]},{"scope_grants":[{"case_id":"case-0","scope_id":None,"assignment_sha256":"bad"}]}])
def test_fixture_rejects_scope_grant_schema(tmp_path: Path, change):
    payload=_fixture_payload(tmp_path); payload.update(change); p=tmp_path/"fixture.json"; p.write_text(json.dumps(payload),encoding="utf-8")
    with pytest.raises(ValueError): load_m2_fixture(p)

def test_fixture_rejects_non_mapping_grant(tmp_path: Path):
    payload=_fixture_payload(tmp_path); payload["scope_grants"]=[None]; p=tmp_path/"fixture.json"; p.write_text(json.dumps(payload),encoding="utf-8")
    with pytest.raises(ValueError): load_m2_fixture(p)

def test_fixture_rejects_dtype_and_shape_payload(tmp_path: Path):
    payload=_fixture_payload(tmp_path); np.save(tmp_path/"inputs.npy",np.zeros((1,1,16),dtype=np.float64)); payload["input"]["dtype"]="<f8"; p=tmp_path/"fixture.json"; p.write_text(json.dumps(payload),encoding="utf-8")
    with pytest.raises(ValueError): load_m2_fixture(p)

def _manifest_payload():
    """生成 manifest 负向测试基础对象。"""
    scopes=[{"scope_id":"standard","parent_scope_ids":[],"expert_ids":["E1"]},{"scope_id":"advanced","parent_scope_ids":["standard"],"expert_ids":["E1","E2"]},{"scope_id":"privileged","parent_scope_ids":["advanced"],"expert_ids":["E1","E2","E3"]}]
    specs=[{"expert_id":e,"kind":"shared" if e=="E0" else "routed","capability_level":"general","scope_ids":[],"architecture_revision":"m2-tiny-v1","weights_sha256":"0"*64,"train_data_scope":"fixture","max_context_length":8,"enabled":True} for e in ("E0","E1","E2","E3")]
    return {"schema_version":1,"execution_config_id":"m2-multi-expert-v1","protocol_id":"m2-scope-lattice-v1","policy":P1_POLICY,"expert_specs":specs,"scopes":scopes,"alpha":1.0,"top_k":1,"router":{"type":"fixed","routed_expert_order":["E1","E2","E3"],"normalize_eps":1e-12},"model":{"d_model":16,"dtype":"float32","device":"cpu"},"fixture":{},"provenance":{"git_commit":"0"*40,"dirty":True}}

@pytest.mark.parametrize("field,value", [("expert_specs",None),("expert_specs",[]),("router",{}),("model",{}),("provenance",{})])
def test_manifest_rejects_each_nested_schema(field,value,tmp_path: Path):
    payload=_manifest_payload(); payload[field]=value; p=tmp_path/"manifest.json"; p.write_text(json.dumps(payload),encoding="utf-8")
    with pytest.raises(ValueError): load_m2_manifest(p,hashlib.sha256(p.read_bytes()).hexdigest())

def test_manifest_rejects_disabled_and_bad_hash_experts(tmp_path: Path):
    for mutation in (lambda x:x.update({"enabled":False}), lambda x:x.update({"weights_sha256":"bad"}), lambda x:x.update({"expert_id":"E9"})):
        payload=_manifest_payload(); mutation(payload["expert_specs"][0]); p=tmp_path/"manifest.json"; p.write_text(json.dumps(payload),encoding="utf-8")
        with pytest.raises(ValueError): load_m2_manifest(p,hashlib.sha256(p.read_bytes()).hexdigest())

def test_expert_aliases_are_constructible():
    assert ExpertE0().expert_id=="E0" and ExpertE1().expert_id=="E1" and ExpertE2().expert_id=="E2" and ExpertE3().expert_id=="E3"

def test_moe_rejects_invalid_coordinator_config_and_hidden_values():
    auth,coord,ctx,reg=_setup(); bad=type("C",(),{"execution_config_id":"other"})()
    with pytest.raises(Exception): M2MoE(bad)
    ass=reg.assignment("case",ctx); route=coord.commit(auth.verify(torch.zeros(2,2),ctx,ass),ctx); model=M2MoE(coord,registry=reg)
    for hidden in (torch.zeros(2,16),torch.zeros(0,1,16),torch.zeros(2,1,15),torch.zeros(2,1,16,dtype=torch.float64),torch.full((2,1,16),float("inf"))):
        with pytest.raises(ValueError): model(hidden,route,ctx)

def test_state_dict_digest_validates_inputs():
    assert len(state_dict_sha256({"w":torch.ones(2,2)}))==64
    for value in ({}, {1:torch.ones(1)}, {"w":torch.sparse_coo_tensor([[0],[0]],[1.],(1,1))}):
        with pytest.raises((ValueError,TypeError)): state_dict_sha256(value)

def test_auth_and_registry_constructor_type_checks():
    with pytest.raises(TypeError): M2AuthExpert(object(),"m2-multi-expert-v1")
    with pytest.raises(TypeError): M2ScopeRegistry(object())

def test_router_rejects_non_tensor_inputs():
    with pytest.raises(ValueError): FixedConstrainedRouter().select(None, None)

def test_router_rejects_nonfinite_logits():
    with pytest.raises(ValueError): FixedConstrainedRouter().select(torch.tensor([[float("nan"),0.,0.]]),torch.tensor([[True,False,False]]))

def test_router_rejects_mask_shape_and_device_contract():
    router=FixedConstrainedRouter(); logits=torch.zeros(1,3)
    with pytest.raises(ValueError): router.select(logits,torch.zeros(2,3,dtype=torch.bool))
    with pytest.raises(ValueError): router.select(logits,None)
