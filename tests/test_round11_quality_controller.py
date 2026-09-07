"""The independent holdout must remain closed when formal accuracy is insufficient."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def controller(monkeypatch,tmp_path,run_count):
    scripts=Path(__file__).resolve().parents[1]/"scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location("round11_postquality_test",scripts/"run_round11_closure_postquality.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,"RESULT",tmp_path)
    monkeypatch.setattr(sys,"argv",["postquality","--gpu","0"])
    report=tmp_path/"reports/formal/summary.json";report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"runs":[{} for _ in range(run_count)],"eligible_arms_for_acceleration":[]}))
    def forbidden(*args,**kwargs):
        raise AssertionError("No training, generation, or processing may run after a failed quality gate")
    monkeypatch.setattr(module,"command",forbidden)
    return module


def test_failed_formal_quality_gate_records_negative_result_without_opening_holdout(monkeypatch,tmp_path):
    module=controller(monkeypatch,tmp_path,9)
    module.main()
    status=json.loads((tmp_path/"postquality/status.json").read_text())
    assert status["state"]=="complete_negative_accuracy_result"
    assert status["test_opened"] is False
    assert not (tmp_path/"postquality/FROZEN.json").exists()


def test_incomplete_seed_comparison_cannot_start_acceleration_or_holdout(monkeypatch,tmp_path):
    module=controller(monkeypatch,tmp_path,3)
    with pytest.raises(RuntimeError,match="all nine"):
        module.main()
    status=json.loads((tmp_path/"postquality/status.json").read_text())
    assert status["state"]=="failed"
    assert status["test_opened"] is False


def test_success_path_benchmarks_then_freezes_before_any_blind_generation(monkeypatch,tmp_path):
    import torch
    module=controller(monkeypatch,tmp_path,9)
    root=tmp_path/"project";monkeypatch.setattr(module,"ROOT",root)
    config=root/"configs/gkeyll/continuum_round11_holdout.json";config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"dataset_root":str(tmp_path/"blind")}))
    def run(name,arm="A",seed=0):
        return {"name":name,"arm":arm,"seed":seed,"checkpoint":str(tmp_path/(name.replace('/','_')+".pt")),
                "late_medians_completed_only":{"field_log10_rmse":.1,"electric_mode1_phase_mae":.1,"perturbation_relative_l2":.1}}
    report={"runs":[run(f"{arm}{seed}",arm,seed) for arm in ("A","B","C") for seed in range(3)],
            "eligible_arms_for_acceleration":["A"],"quality_configuration":{}}
    (tmp_path/"reports/formal/summary.json").write_text(json.dumps(report))
    monkeypatch.setattr(torch,"load",lambda *a,**kw:{"config":{"width":128}})
    monkeypatch.setattr(module,"train_candidate",lambda name,*a,**kw:run(name))
    monkeypatch.setattr(module,"acceptance",lambda *a:{"passed":True})
    monkeypatch.setattr(module,"no_material_degradation",lambda *a:{"passed":True})
    monkeypatch.setattr(module,"sha",lambda *a:"test-hash")
    events=[]
    def benchmark(candidates,*a):
        assert not (tmp_path/"postquality/FROZEN.json").exists()
        events.append("benchmark")
        return candidates[0]
    def generation(args,*a):
        assert "blind" in args
        assert (tmp_path/"postquality/FROZEN.json").exists()
        assert events==["benchmark"]
        events.append("blind_generation")
        raise RuntimeError("test stops at generation boundary")
    monkeypatch.setattr(module,"benchmark_before_freeze",benchmark)
    monkeypatch.setattr(module,"command",generation)
    with pytest.raises(RuntimeError,match="generation boundary"):
        module.main()
    assert events==["benchmark","blind_generation"]
