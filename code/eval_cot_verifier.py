import json
import os
from collections import defaultdict


def _pct(num, den):
    return (num / den) * 100 if den else 0.0


def _empty_counts():
    return {"total": 0, "accept": 0, "reject": 0}


def _bucket_for(raw_type, sub_type):
    if raw_type == "type1_golden" or sub_type == "type1_correct":
        return "T1"
    if raw_type == "type2_spurious" or sub_type.startswith("type2_"):
        return "T2"
    if raw_type == "type3_fully_wrong" or sub_type == "type3_fully_wrong":
        return "T3"
    if raw_type == "type4_calc_error" or sub_type == "type4_calc_error":
        return "T4_H"
    if raw_type in {"type4_answer_perturb", "type4_inconsistent_easy"}:
        return "T4_E"
    if sub_type in {"type4_answer_perturb", "type4_inconsistent_easy"}:
        return "T4_E"
    return "UNKNOWN"


def evaluate_verifier_metrics(result_file):
    """
    按 CLAUDE.md / paper Table 1 口径计算 TrustTable 指标:
    1. VCAR = T1_acc / |T1|
    2. DIR_spur = T2_rej / |T2|，T2 为三个 subtype 的 raw-count union
    3. DIR_inc_H = T4_calc_rej / |T4_calc|，Table 1 只用 Hard T4
    4. FP_H = T1_acc / (T1_acc + T2_acc + T3_acc + T4_calc_acc)

    注意: type4_answer_perturb 是 Easy T4，只单独报告，不混入 Table 1 主指标。
    """
    if not os.path.exists(result_file):
        print(f"Error: File {result_file} not found.")
        return

    print(f"Loading results from: {result_file}")
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    stats = defaultdict(_empty_counts)
    subtype_stats = defaultdict(_empty_counts)

    for item in results:
        raw_type = item.get("target_type", "unknown")
        sub_type = item.get("specific_subtype", "unknown")
        decision = item.get("verifier_decision", "UNKNOWN").upper()
        bucket = _bucket_for(raw_type, sub_type)

        stats[bucket]["total"] += 1
        subtype_stats[sub_type]["total"] += 1

        if decision == "ACCEPT":
            stats[bucket]["accept"] += 1
            subtype_stats[sub_type]["accept"] += 1
        elif decision == "REJECT":
            stats[bucket]["reject"] += 1
            subtype_stats[sub_type]["reject"] += 1

    t1_stats = stats["T1"]
    t2_stats = stats["T2"]
    t3_stats = stats["T3"]
    t4h_stats = stats["T4_H"]
    t4e_stats = stats["T4_E"]

    vcar = _pct(t1_stats["accept"], t1_stats["total"])
    dir_spur = _pct(t2_stats["reject"], t2_stats["total"])
    dir_inc_h = _pct(t4h_stats["reject"], t4h_stats["total"])

    fp_h_den = (
        t1_stats["accept"]
        + t2_stats["accept"]
        + t3_stats["accept"]
        + t4h_stats["accept"]
    )
    fp_h = _pct(t1_stats["accept"], fp_h_den)
    dir_inc_easy = _pct(t4e_stats["reject"], t4e_stats["total"])

    fp_easy_den = (
        t1_stats["accept"]
        + t2_stats["accept"]
        + t3_stats["accept"]
        + t4e_stats["accept"]
    )
    fp_easy = _pct(t1_stats["accept"], fp_easy_den)

    print("\n" + "="*80)
    print("CLAUDE.md / Paper Table 1 metrics")
    print(f"{'METRIC':<25} | {'VALUE':<10} | {'DEFINITION'}")
    print("-" * 80)

    print(f"{'VCAR':<25} | {vcar:5.1f}%     | T1_acc / |T1|")
    print(f"{'DIR_spur':<25} | {dir_spur:5.1f}%     | T2_rej / |T2 union|")
    print(f"{'DIR_inc_H':<25} | {dir_inc_h:5.1f}%     | type4_calc_error_rej / |type4_calc_error|")
    print(f"{'FP_H':<25} | {fp_h:5.1f}%     | T1_acc / (T1+T2+T3+T4_H accepted)")
    print("-" * 80)
    print("Appendix / non-Table-1 Easy T4:")
    print(f"{'DIR_inc_E':<25} | {dir_inc_easy:5.1f}%     | type4_answer_perturb_rej / |type4_answer_perturb|")
    print(f"{'FP_E':<25} | {fp_easy:5.1f}%     | T1_acc / (T1+T2+T3+T4_E accepted)")
    print("-" * 80)
    print(">>> Bucket Counts:")
    print(f"{'Bucket':<10} | {'Total':<6} | {'Accepted':<8} | {'Rejected':<8} | {'Unknown':<8}")
    print("-" * 80)
    for bucket in ["T1", "T2", "T3", "T4_H", "T4_E", "UNKNOWN"]:
        data = stats[bucket]
        if data["total"] == 0:
            continue
        unknown = data["total"] - data["accept"] - data["reject"]
        print(
            f"{bucket:<10} | {data['total']:<6} | {data['accept']:<8} | "
            f"{data['reject']:<8} | {unknown:<8}"
        )
    print("-" * 80)
    print(">>> Detailed Breakdown by Subtype:")
    print(f"{'Subtype':<30} | {'Total':<6} | {'Accepted':<8} | {'Rejected':<8} | {'Unknown':<8} | {'DIR (%)'}")
    print("-" * 80)

    sorted_subtypes = sorted(subtype_stats.items(), key=lambda x: x[0])
    for sub, data in sorted_subtypes:
        total = data["total"]
        if total == 0:
            continue
        d_rate = _pct(data["reject"], total)
        unknown = total - data["accept"] - data["reject"]
        print(f"{sub:<30} | {total:<6} | {data['accept']:<8} | {data['reject']:<8} | {unknown:<8} | {d_rate:5.1f}%")

    print("="*80 + "\n")

    return {
        "VCAR": vcar,
        "DIR_spur": dir_spur,
        "DIR_inc_H": dir_inc_h,
        "FP_H": fp_h,
        "DIR_inc_E": dir_inc_easy,
        "FP_E": fp_easy,
    }

if __name__ == "__main__":
    import sys
    # Pass a result file on the command line; otherwise use the default output path.
    if len(sys.argv) >= 2:
        RESULT_FILE = sys.argv[1]
    else:
        RESULT_FILE = "../outputs/baseline_cot_verifier_results.json"
    evaluate_verifier_metrics(RESULT_FILE)
