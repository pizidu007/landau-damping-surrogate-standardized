"""Maintain a concise local status page while the authorized experiment runs."""
from datetime import datetime,timezone
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]/"results/continuum_v1_closure_round11"


def read(path,default=None):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default


def process_alive(status,script):
    try:
        arguments=Path(f"/proc/{int(status['pid'])}/cmdline").read_bytes().split(b"\0")
        return any(Path(arg.decode(errors="replace")).name==Path(script).name for arg in arguments if arg)
    except (KeyError,ValueError,FileNotFoundError,ProcessLookupError):
        return False


def update():
    control=read(ROOT/"control/status.json",{})
    post=read(ROOT/"postquality/status.json",{})
    current=control.get("current_task")
    stage=control.get("state","unknown")
    main_alive=process_alive(control,"scripts/run_round11_closure_pipeline.py")
    post_alive=process_alive(post,"scripts/run_round11_closure_postquality.py")
    if stage in ("starting","running") and not main_alive:
        stage="interrupted (状态文件未收尾，主进程已不存在)"
    post_stage=post.get("state","未启动")
    if post_stage not in ("complete","complete_negative_accuracy_result","failed","未启动") and not post_alive:
        post_stage=f"interrupted (原状态 {post_stage}，后续进程已不存在)"
    lines=["# Round 11 执行进度","",f"更新时间：{datetime.now(timezone.utc).isoformat()}（UTC）", "",
           f"主实验状态：`{stage}`；当前任务：`{current}`。",
           f"后续流程状态：`{post_stage}`；独立盲测已打开：`{post.get('test_opened',False)}`。",
           f"进程存活核对：主实验 {main_alive}；后续流程 {post_alive}。", ""]
    if current:
        path=ROOT/current/"run.log"
        if path.exists():
            for line in reversed(path.read_text(errors="replace").splitlines()[-15:]):
                try:event=json.loads(line)
                except json.JSONDecodeError:continue
                if "validation_time" in event:
                    lines.append(f"最新事件：候选模型自由验证到 t={event['validation_time']:g}，累计失效 {event['failed']} 条。")
                elif "batch" in event:
                    lines.append(f"最新事件：闭环训练 epoch {event['epoch']}，batch {event['batch']}/{event['batches']}，连续梯度时长 {event['horizon']:g}。")
                elif "epoch" in event:
                    lines.append(f"最新事件：{event.get('phase')} epoch {event['epoch']} 完成。")
                break
    gate=read(ROOT/"oracle_gate_formal_mode16.json",{})
    if gate.get("all_validation_check"):
        check=gate["all_validation_check"]["summary"]
        lines += ["",f"真实热流 oracle：{check['complete']}/{check['case_count']} 条完成 t=80；未滤波扰动状态最大相对误差 {check['max_unfiltered_perturbation_error']:.3%}。"]
    lines += ["", "下表为已完成实验。误差中位数仅含完成案例，不能跨不同幸存集合直接认定模型更好；配对比较见 reports 下的 summary.json。", "",
              "| 实验 | 完成 t=80 | 全程 log10 场能 RMSE | 全程相位 MAE | 扰动状态相对 L2 |", "|---|---:|---:|---:|---:|"]
    paths=sorted((ROOT/"pilot").glob("*/summary.json"))+sorted((ROOT/"formal").glob("*/*/summary.json"))
    def fmt(value):return "—" if value is None else f"{value:.4g}"
    for path in paths:
        value=read(path,{})
        summary=value.get("full_validation")
        if summary:
            lines.append(f"| {path.parent.relative_to(ROOT)} | {summary['complete']}/{summary['case_count']} | {fmt(summary.get('full_t80_field_energy_log10_rmse_median'))} | {fmt(summary.get('electric_mode1_phase_mae_median'))} | {fmt(summary.get('perturbation_relative_l2_median'))} |")
    lines += ["", "执行顺序：oracle → 历史跨度 pilot 与 A/B/C → 三个训练随机种子 → 精度门 → 闭合调用频率与网络宽度 → 冻结 → 新增独立参数盲测与计时。", "",
              "正式模型采用 mode 16、dt=0.02、反射等变约束；A/B/C 使用一致设置。独立盲测通过精度门后才生成；未通过则保留负结果。"]
    for name,value in (("主实验",control),("后续流程",post)):
        if value.get("error"):lines += ["",f"{name}错误：`{value['error']}`。"]
    temporary=ROOT/"RUN_STATUS.tmp";temporary.write_text("\n".join(lines)+"\n");temporary.replace(ROOT/"RUN_STATUS.md")
    return post.get("state") in ("complete","complete_negative_accuracy_result","failed")


if __name__=="__main__":
    while True:
        if update():break
        time.sleep(30)
