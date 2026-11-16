"""Export audited study reports and source snapshots into the existing NTUT repository.

--preview writes only runs/comparison/README.preview.md and permits pending runs.
The normal export refuses to publish incomplete low-data results.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import analyze as A
from low_data_status import audit


def section(text, heading, following, body):
    start, end = text.index(heading), text.index(following)
    return text[:start] + heading + "\n\n" + body.strip() + "\n\n" + text[end:]


def readme(text, rep, strict, hits):
    done = sum(r["complete"] for r in rep["low_data_status"].values())
    text = re.sub(r"^> 狀態：.*$", f"> 狀態：2026-11-16 統計與定位指標修訂；少量資料完整 run {done}/9。完成狀態見 [low_data_status.json](low_data_status.json)。", text, count=1, flags=re.M)
    conclusion = A.main_conclusions(rep).split("\n", 1)[1].strip()
    h = hits["uclm:test"]["gate/cal"]["iou_0.50"]
    fa = strict["uclm:test"]["WeVisOne"]["gate"]["iou_0.50"]["f1"]
    fb = strict["uclm:test"]["YOLO26m-seg"]["cal"]["iou_0.50"]["f1"]
    conclusion += (f"\n\n**嚴格定位補充**：UCLM held-out test、固定 gate/cal、union-mask IoU ≥ 0.50 的合併 F1 "
                   f"為 WeVisOne {fa:.3f}、YOLO {fb:.3f}；相同 seed 共同命中影像上的配對 Dice 差為 "
                   f"{A.fmt(h['dice_diff'])}（mean ± SD）。此條件式差距與全病灶 Dice 差不能直接相減來估計定位的因果貢獻。")
    conclusion += "\n\n其他結果：BUSI 內部差距較小；框提示與混合實驗見第 6–7 節；強增強的取捨見第 10 節；多來源加入 UCLM 訓練病人後，YOLO 與 WeVisOne 的差距縮小，且 WeVisOne 閘門可能刪掉大量病灶（第 16 節）。預訓練曝光仍無法排除（第 19 節）。"
    text = section(text, "## 結論", "## 1. 資料與切分", conclusion)

    body = ["這一節包含兩種不同的分析；固定操作點的 CI 不適用於測試集選出的 matched-FPR 包絡線。", "",
            "### 測試集選門檻的操作曲線", "",
            "逐模型、逐 seed、逐測試集掃分數門檻，在測試 FPR ≤ 5%／10%／15% 的限制下，取測試病灶 Dice 最高的點。"
            "選擇同時使用測試正常標籤及病灶 GT，因此可能偏樂觀，只作描述性分析。相同的是 FPR 上限，各模型實際 FPR 未必相等。"
            "mean ± SD 是訓練 seed 間的變異，不是信賴區間；本分析沒有 matched-FPR 優勢的顯著性檢定。", "",
            "| 資料集 | 模型 | FPR ≤ 5%：Dice / 零重疊 % | FPR ≤ 10% | FPR ≤ 15% |", "|---|---|---|---|---|"]
    for spec, models in rep["matched_fpr"].items():
        for model, values in models.items():
            cells = [f"{A.fmt(values[str(t)]['lesion_dice'])} / {A.fmt(values[str(t)]['zero_overlap'], True)}" for t in (0.05, 0.10, 0.15)]
            body.append(f"| {spec} | {model} | " + " | ".join(cells) + " |")
    body += ["", "各點實際 FPR 與所選門檻見 [report.json](report.json) 的 `matched_fpr` 欄位。", "",
             "![測試集操作曲線](fig_operating_uclm.png)", "",
             "### 固定操作點的病人配對 bootstrap", "",
             "比較驗證集校準／預設門檻的固定預測；先對每張影像跨 seed 平均，再以病人為單位配對重抽 5,000 次。"
             "不在重抽樣內重新訓練或校準門檻。95% percentile CI 未作多重比較校正，只涵蓋固定模型與門檻下的病人抽樣不確定性。"
             "UCLM 使用病人 ID；BUSI 與 BUS_UC 沒有病人 ID，以既有影像／去重群組作抽樣單位，不能將群組數當成已知病人數。ΔFPR > 0 對 WeVisOne 不利，ΔDice > 0 有利。", "",
             "| 比較 | 影像 / 病人或群組 | ΔFPR 百分點 [95% CI] | Δ病灶 Dice [95% CI] |", "|---|---:|---:|---:|"]
    for label, b in rep["paired_bootstrap"].items():
        fpr = "–" if b["fpr_diff"] != b["fpr_diff"] else f"{100*b['fpr_diff']:+.1f} [{100*b['fpr_diff_ci'][0]:+.1f}, {100*b['fpr_diff_ci'][1]:+.1f}]"
        body.append(f"| {label.replace('|', '/')} | {b['n_images']} / {b['n_patients']} | {fpr} | "
                    f"{b['lesion_dice_diff']:+.3f} [{b['lesion_dice_diff_ci'][0]:+.3f}, {b['lesion_dice_diff_ci'][1]:+.3f}] |")
    body += ["", "UCLM held-out test 的 gate/cal FPR 差 CI 跨 0，無法判定 FPR 有差異；UCLM all 的 FPR 差 CI 全為正，WeVisOne FPR 較高。"
             "強增強版的 FPR 差 CI 跨 0 也不能解讀成兩模型等效。UCLM test 是 all 的子集，兩列不是獨立驗證。"]
    text = section(text, "## 5. 在相同誤報率下比較病灶表現", "## 6. 端到端失敗分解", "\n".join(body))

    body = ["保留任意 GT 像素重疊的寬鬆 F1，新增 **union-mask IoU ≥ 0.25／0.50** 的嚴格定位 F1（256×256）。"
            "TP 為命中病灶影像；未命中病灶計 FN；有非空但未命中的病灶預測另計 FP；正常影像有任一前景像素也計 FP。"
            "門檻等號包含在命中條件內。這是影像級 union-mask 指標，並非病灶實例的一對一匹配 F1；"
            "同一張影像額外預測的區域會降低 union IoU，但不逐實例加計 FP。", "",
            "以下採各自驗證集固定門檻（WeVisOne gate／YOLO cal）；F1 為三 seed 計數合併值，括號為 TP／FP／FN。"
            "三 seed 反覆使用相同測試影像，不能當成三倍獨立樣本。", "",
            "| 資料集 | 命中條件 | WeVisOne gate F1 (TP/FP/FN) | YOLO cal F1 (TP/FP/FN) |", "|---|---|---:|---:|"]
    for spec in A.TESTS:
        for criterion in ("any_overlap", "iou_0.25", "iou_0.50"):
            cells = []
            for name, op in (("WeVisOne", "gate"), ("YOLO26m-seg", "cal")):
                r = strict[spec][name][op][criterion]
                cells.append(f"{r['f1']:.3f} ({r['tp']}/{r['fp']}/{r['fn']})")
            body.append(f"| {spec} | {criterion} | " + " | ".join(cells) + " |")
    body += ["", "完整模型、raw／校準操作點、precision／recall 與跨 seed F1 的 mean ± SD 見 [STRICT_F1.md](STRICT_F1.md)；舊指標見 [F1.md](F1.md)。", "",
             "### 相同 seed 共同命中的病灶影像", "",
             "A = WeVisOne gate，B = YOLO cal。同 seed 按影像 ID 配對，僅對雙方共同命中的影像比較 Dice。"
             "下列計數為 seed 合併，Dice 先逐 seed 在共同命中影像上平均，再跨 seed 平均。"
             "不同 seed／命中條件的共同子集會改變；這是模型依賴的條件式比較，不能單獨證明無條件的分割優勢或 decoder 因果貢獻。", "",
             "| 資料集 | 命中條件 | 共同 / 僅 A / 僅 B / 皆未 | A Dice | B Dice | 配對 ΔDice mean ± SD |", "|---|---|---|---:|---:|---:|"]
    for spec in ("busi:test", "uclm:test", "uclm:all"):
        for criterion in ("any_overlap", "iou_0.25", "iou_0.50"):
            r = hits[spec]["gate/cal"][criterion]
            parts = " / ".join(str(r[k]) for k in ("both", "only_a", "only_b", "neither"))
            body.append(f"| {spec} | {criterion} | {parts} | {A.fmt(r['dice_a'])} | {A.fmt(r['dice_b'])} | {A.fmt(r['dice_diff'])} |")
    body += ["", "所有測試集與 raw/0.25 配對另見 [COMMON_HITS.md](COMMON_HITS.md)；每個 seed 的四組影像 ID 與 Dice 保存在 [common_hits.json](common_hits.json)。"]
    text = section(text, "## 12. 影像層級 F1", "## 13. 記憶測試：輕微幾何變換後表現是否下降", "\n".join(body))

    body = ["每個比例使用相對應的子集 seed 與訓練 seed 42／43／44；同一 seed 的較小子集包含在較大子集中。"
            "WeVisOne 維持 230 次更新、5 次驗證，並以 BUSI 驗證集選 checkpoint 與 gate；YOLO 使用原有訓練設定。"
            "mean ± sample SD 同時包含子集抽樣與訓練 seed 的變異，未將兩者分離。以下都是固定操作點的全病灶影像平均 Dice。", "",
            "| 訓練比例 | WeVisOne runs | YOLO runs | WeVisOne BUSI Dice | YOLO BUSI Dice | WeVisOne UCLM Dice | YOLO UCLM Dice | WeVisOne UCLM AUROC | YOLO UCLM AUROC |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for f, models in rep["low_data"].items():
        a, b = models["wevis"], models["yolo"]
        if not a or not b:
            continue
        am, bm = a["metrics"], b["metrics"]
        vals = [am["busi:test"]["gate"]["lesion_dice"], bm["busi:test"]["cal"]["lesion_dice"],
                am["uclm:all"]["gate"]["lesion_dice"], bm["uclm:all"]["cal"]["lesion_dice"],
                am["uclm:all"]["auroc"], bm["uclm:all"]["auroc"]]
        body.append(f"| {100*float(f):g}% | {a['n_runs']} | {b['n_runs']} | " + " | ".join(A.fmt(v) for v in vals) + " |")
    body += ["", f"少量資料完整 run：{done}/9；只有 230 步、5 次驗證、checkpoint 與所有驗證／測試影像 ID 均完整的 run 才列入。"
             "逐 run 檢查見 [low_data_status.json](low_data_status.json)。預訓練曝光未知，少資料優勢不能單獨歸因於架構或訓練方法。"]
    text = section(text, "## 15. 少量資料（BUSI 訓練集的 10% / 25% / 50%）", "## 16. 多來源訓練", "\n".join(body))
    text = text.replace("少量資料、交叉驗證的 WeVisOne 都只有 1 個 seed；多來源已補到 3 個 seeds。",
                        f"少量資料目標為每個比例 3 seeds，目前通過完整檢查 {done}/9；多來源已有 3 seeds；交叉驗證每折仍只有 1 個 seed。")
    text = text.replace("只要框打中（hit），兩個模型的 Dice 都在 0.88–0.91，mask 品質差不多。",
                        "各模型自行命中的影像 Dice 約 0.88–0.91，但這兩群影像不一定相同；不能據此判定 mask 品質相同。配對共同命中比較見第 12 節。")
    text = text.replace("WeVisOne 的外部優勢來自語言模型的框與有無病灶判斷，不是 decoder。",
                        "框與有無病灶判斷是外部差距的重要來源；混合與 oracle 實驗不足以證明 decoder 完全沒有貢獻，共同命中的配對比較見第 12 節。")
    text = text.replace("WeVisOne 的外部優勢來自語言模型產生的框與 `None` 判斷，不是 decoder。",
                        "框與 `None` 判斷是外部差距的重要來源；共同命中的條件式 Dice 仍有差異，不能推論 decoder 完全沒有貢獻（第 12 節）。")
    text = text.replace("- 只有病灶的資料集（BUS_UC、BUS-BRA、BrEaST）沒有正常影像，", "- BUS_UC、BUS-BRA 沒有正常影像，BrEaST 有 4 張正常影像，")
    if "| [SUMMARY.md](SUMMARY.md) |" not in text:
        text = text.replace("| [F1.md](F1.md) / [f1.json](f1.json) |", "| [SUMMARY.md](SUMMARY.md) | 區分固定門檻推論與測試集操作曲線的主要統計結論 |\n| [STRICT_F1.md](STRICT_F1.md) / [strict_f1.json](strict_f1.json) | IoU 0.25／0.50 的嚴格影像級定位 F1 |\n| [COMMON_HITS.md](COMMON_HITS.md) / [common_hits.json](common_hits.json) | 同 seed 共同命中配對 Dice 與四組影像 ID |\n| [low_data_status.json](low_data_status.json) | 9 個少量資料 run 的完整性檢查 |\n| [F1.md](F1.md) / [f1.json](f1.json) |")
    note = "少量資料補跑與完整性檢查：`powershell -File scripts/queue_master.ps1 -LowDataOnly`；`python scripts/low_data_status.py`。"
    if note not in text:
        text += "\n" + note + "\n"
    layout = "`scripts/` 是原始訓練工作區的程式快照。重現時使用 `<workspace>/scripts/`，並將本 repo 放在 `<workspace>/Medical-image-NTUT/`、模型程式放在 `<workspace>/WeVisOne-VL/`；資料、runs 與外部資料路徑沿用程式設定。不要直接在本報告子目錄執行訓練指令。"
    if layout not in text:
        text += "\n" + layout + "\n"
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    rep = json.loads((A.OUT / "report.json").read_text(encoding="utf-8"))
    strict = json.loads((A.OUT / "strict_f1.json").read_text(encoding="utf-8"))
    hits = json.loads((A.OUT / "common_hits.json").read_text(encoding="utf-8"))
    status = audit()
    if args.preview:
        rep.setdefault("low_data_status", status)
    if not args.preview:
        if not all(r["complete"] for r in status.values()) or status != rep["low_data_status"]:
            raise RuntimeError("Low-data runs incomplete or report stale; regenerate analysis before export")
        for f in ("0.1", "0.25", "0.5", "1.0"):
            assert rep["low_data"][f]["wevis"]["n_runs"] == 3
    dest = A.common.NTUT / "results" / "wevisone_vs_yolo"
    target = readme((dest / "README.md").read_text(encoding="utf-8"), rep, strict, hits)
    if args.preview:
        (A.OUT / "README.preview.md").write_text(target, encoding="utf-8")
        print("Preview generated; repository unchanged")
        return
    (dest / "README.md").write_text(target, encoding="utf-8")
    files = ["REPORT.md", "report.json", "SUMMARY.md", "F1.md", "f1.json", "STRICT_F1.md", "strict_f1.json",
             "COMMON_HITS.md", "common_hits.json", "low_data_status.json", "fig_operating_uclm.png"]
    for name in files:
        shutil.copy2(A.OUT / name, dest / name)
    scripts = ["analyze.py", "f1_table.py", "low_data_status.py", "queue_master.ps1", "test_localization.py",
               "fpr_metrics.py", "presence_experiment.py", "test_fpr.py", "sync_study.py", "test_study_export.py", "complete_low_data.ps1"]
    for name in scripts:
        shutil.copy2(Path(__file__).parent / name, dest / "scripts" / name)
    manifest = [str(p.relative_to(A.common.NTUT)).replace("\\", "/")
                for p in [dest / "README.md"] + [dest / n for n in files] + [dest / "scripts" / n for n in scripts]]
    (A.OUT / "publish_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Exported {len(manifest)} audited files")


if __name__ == "__main__":
    main()
