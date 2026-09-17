"""M2 manifest 的严格解析和冻结配置校验。"""
from __future__ import annotations
import hashlib,hmac,json,re
from pathlib import Path
from typing import Any,Dict,List
from .scope import validate_scope_lattice
from .router import P1_POLICY,P2_POLICY
_HASH=re.compile(r"^[0-9a-f]{64}$")
def load_m2_manifest(path:Path,expected_sha256:str)->Dict[str,Any]:
    """校验 manifest 完整摘要、字段集合、scope lattice 和冻结参数。"""
    if not isinstance(path,Path) or not path.is_file(): raise FileNotFoundError("manifest 不存在")
    raw=path.read_bytes()
    if not isinstance(expected_sha256,str) or not hmac.compare_digest(hashlib.sha256(raw).hexdigest(),expected_sha256): raise ValueError("M2A_MANIFEST_DIGEST_MISMATCH")
    def dup(pairs:List[Any])->Dict[str,Any]:
        """拒绝重复 JSON 字段。"""
        out={}
        for k,v in pairs:
            if k in out: raise ValueError("manifest 字段重复")
            out[k]=v
        return out
    payload=json.loads(raw.decode("utf-8"),object_pairs_hook=dup,parse_constant=lambda x: (_ for _ in ()).throw(ValueError("非有限常量")))
    required={"schema_version","execution_config_id","protocol_id","policy","expert_specs","scopes","alpha","top_k","router","model","fixture","provenance"}
    if not isinstance(payload,dict) or set(payload)!=required: raise ValueError("M2A_ARTIFACT_SCHEMA_MISMATCH")
    if payload["schema_version"]!=1 or payload["execution_config_id"]!="m2-multi-expert-v1" or payload["protocol_id"]!="m2-scope-lattice-v1" or payload["policy"] not in {P1_POLICY,P2_POLICY} or payload["alpha"]!=1.0 or payload["top_k"]!=1: raise ValueError("M2 manifest 冻结字段不匹配")
    specs=payload["expert_specs"]
    if not isinstance(specs,list) or {x.get("expert_id") for x in specs if isinstance(x,dict)}!={"E0","E1","E2","E3"}: raise ValueError("expert_specs 必须包含 E0-E3")
    expert_keys={"expert_id","kind","capability_level","scope_ids","architecture_revision","weights_sha256","train_data_scope","max_context_length","enabled"}
    if len(specs)!=4 or any(not isinstance(x,dict) or set(x)!=expert_keys for x in specs): raise ValueError("ExpertSpec schema 非法")
    for x in specs:
        if not x.get("enabled",False) or not isinstance(x["weights_sha256"],str) or _HASH.fullmatch(x["weights_sha256"]) is None: raise ValueError("ExpertSpec 非法")
    validate_scope_lattice(payload["scopes"],[x["expert_id"] for x in specs if x.get("enabled")])
    if payload["router"]!={"type":"fixed","routed_expert_order":["E1","E2","E3"],"normalize_eps":1e-12}: raise ValueError("router 配置不匹配")
    if payload["model"]!={"d_model":16,"dtype":"float32","device":"cpu"}: raise ValueError("model 配置不匹配")
    if not isinstance(payload["provenance"],dict) or set(payload["provenance"])!={"git_commit","dirty"} or not isinstance(payload["provenance"]["git_commit"],str) or re.fullmatch(r"[0-9a-f]{40}",payload["provenance"]["git_commit"]) is None or not isinstance(payload["provenance"]["dirty"],bool): raise ValueError("provenance 配置不匹配")
    return payload
