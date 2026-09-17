"""M2 fixture、ledger、summary 的最小严格校验工具。"""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
from typing import Any,Mapping
import torch
import numpy as np
from .types import M2CallLedger
_HASH=re.compile(r"^[0-9a-f]{64}$")
def state_dict_sha256(state_dict:Mapping[str,torch.Tensor])->str:
    """按稳定 key/shape/dtype/bytes 编码计算 Expert 权重摘要。"""
    if not isinstance(state_dict,Mapping) or not state_dict: raise ValueError("state_dict 必须非空")
    digest=hashlib.sha256()
    for key in sorted(state_dict):
        value=state_dict[key]
        if not isinstance(key,str) or not isinstance(value,torch.Tensor) or value.is_sparse: raise TypeError("state_dict 仅允许 dense Tensor")
        array=value.detach().cpu().contiguous().numpy(); little=np.asarray(array,dtype=array.dtype.newbyteorder("<"),order="C")
        digest.update(key.encode("utf-8")); digest.update(str(little.dtype).encode("ascii")); digest.update(str(tuple(little.shape)).encode("ascii")); digest.update(little.tobytes())
    return digest.hexdigest()
def _digest(path:Path)->str:
    """计算文件 SHA-256。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()
def load_m2_fixture(path:Path)->dict[str,Any]:
    """加载 fixture 并核对 little-endian NPY 摘要。"""
    if not isinstance(path,Path) or not path.is_file(): raise FileNotFoundError("fixture 不存在")
    payload=json.loads(path.read_text(encoding="utf-8"))
    req={"schema_version","seeds","input","cases","reference","expected","scope_grants"}
    if not isinstance(payload,dict) or set(payload)!=req or payload["schema_version"]!=1: raise ValueError("M2A_ARTIFACT_SCHEMA_MISMATCH")
    for section,name in (("input","inputs.npy"),("reference","reference_outputs.npy")):
        item=payload[section]; target=path.parent/name
        if item.get("path")!=name or item.get("dtype")!="<f4" or not target.is_file() or item.get("sha256")!=_digest(target): raise ValueError("fixture NPY 摘要/格式不匹配")
        arr=np.load(target,allow_pickle=False)
        if arr.dtype!=np.dtype("<f4") or not arr.flags.c_contiguous or list(arr.shape)!=item.get("shape"): raise ValueError("fixture NPY shape/dtype 不匹配")
    grants=payload["scope_grants"]
    if not isinstance(grants,list) or not isinstance(payload["cases"],list): raise ValueError("scope_grants/cases 必须为数组")
    if any(not isinstance(x,dict) for x in grants): raise ValueError("scope_grants schema 非法")
    if {x.get("case_id") for x in grants}!={x.get("case_id") for x in payload["cases"]}: raise ValueError("scope_grants 与 cases 不一致")
    if any(set(x)!={"case_id","scope_id","assignment_sha256"} or (x["scope_id"] is not None and x["scope_id"] not in {"standard","advanced","privileged"}) or not isinstance(x["assignment_sha256"],str) or not _HASH.fullmatch(x["assignment_sha256"]) for x in grants): raise ValueError("scope_grants schema 非法")
    return payload
def validate_m2_call_ledger(payload:Mapping[str,Any])->None:
    """验证台账事件顺序、索引和一次调用计数。"""
    if not isinstance(payload,Mapping) or set(payload)!={"schema_version","run_id","execution_config_id","policy","events"} or payload["schema_version"]!=1: raise ValueError("台账 schema 非法")
    for i,event in enumerate(payload["events"]):
        if set(event)!={"sequence","case_id","stage","expert_id","kind","batch_indices","count"} or event["sequence"]!=i or event["count"]!=1 or not event["batch_indices"]: raise ValueError("台账事件非法")
def validate_m2_summary(payload:Mapping[str,Any])->None:
    """验证 M2 summary 的状态和退出码字段。"""
    req={"schema_version","run_id","execution_config_id","protocol_id","policy","status","exit_code","cases","metrics","determinism","coverage","artifacts","provenance","failure"}
    if not isinstance(payload,Mapping) or set(payload)!=req or payload["schema_version"]!=1 or payload["status"] not in {"complete","failed","not_run"} or not isinstance(payload["exit_code"],int): raise ValueError("summary schema 非法")
    if not isinstance(payload["cases"],list) or not isinstance(payload["metrics"],Mapping) or not isinstance(payload["determinism"],Mapping) or not isinstance(payload["coverage"],Mapping) or not isinstance(payload["artifacts"],Mapping) or not isinstance(payload["provenance"],Mapping): raise ValueError("summary nested schema 非法")
    if payload["failure"] is not None and (not isinstance(payload["failure"],Mapping) or set(payload["failure"])!={"code","stage","case_id","retryable"} or payload["failure"]["retryable"] is not False): raise ValueError("summary failure schema 非法")
