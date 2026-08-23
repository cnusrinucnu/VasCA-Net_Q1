#!/usr/bin/env python3
"""
Parse VasCA-Net training logs (logs_*.txt) into a single summary CSV/table.
"""
import re
import sys
import glob
import csv

EPOCH_RE = re.compile(
    r"train_loss=(?P<train_loss>[\d.]+)\s+"
    r"val_loss=(?P<val_loss>[\d.]+)\s+"
    r"Se=(?P<se>[\d.]+)\s+"
    r"Sp=(?P<sp>[\d.]+)\s+"
    r"F1=(?P<f1>[\d.]+)\s+"
    r"ACC=(?P<acc>[\d.]+)\s+"
    r"AUC=(?P<auc>[\d.]+)\s+"
    r"\((?P<time>[\d.]+)s\)"
)
COMPLETE_RE = re.compile(r"Training complete\. Best val F1: ([\d.]+)")
EARLY_STOP_RE = re.compile(r"Early stopping at epoch (\d+)")


def parse_file(path):
    rows = []
    epoch_num = 0
    status = "in_progress"
    final_best_f1 = None
    early_stop_epoch = None

    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()

    for line in lines:
        m = EPOCH_RE.search(line)
        if m:
            epoch_num += 1
            rows.append({
                "file": path,
                "epoch": epoch_num,
                "train_loss": float(m.group("train_loss")),
                "val_loss": float(m.group("val_loss")),
                "Se": float(m.group("se")),
                "Sp": float(m.group("sp")),
                "F1": float(m.group("f1")),
                "ACC": float(m.group("acc")),
                "AUC": float(m.group("auc")),
                "epoch_time_s": float(m.group("time")),
            })
        cm = COMPLETE_RE.search(line)
        if cm:
            status = "complete"
            final_best_f1 = float(cm.group(1))
        em = EARLY_STOP_RE.search(line)
        if em:
            early_stop_epoch = int(em.group(1))

    return rows, status, final_best_f1, early_stop_epoch


def main():
    files = sys.argv[1:] if len(sys.argv) > 1 else sorted(glob.glob("logs_*.txt"))
    if not files:
        print("No log files found or specified.")
        return

    all_rows = []
    summary = []

    for path in files:
        rows, status, final_best_f1, early_stop_epoch = parse_file(path)
        all_rows.extend(rows)

        best_row = max(rows, key=lambda r: r["F1"]) if rows else None
        last_row = rows[-1] if rows else None

        summary.append({
            "file": path,
            "status": status,
            "epochs_logged": len(rows),
            "early_stop_epoch": early_stop_epoch,
            "reported_best_F1": final_best_f1,
            "best_F1_seen": best_row["F1"] if best_row else None,
            "best_F1_epoch": best_row["epoch"] if best_row else None,
            "best_row_Se": best_row["Se"] if best_row else None,
            "best_row_Sp": best_row["Sp"] if best_row else None,
            "best_row_ACC": best_row["ACC"] if best_row else None,
            "best_row_AUC": best_row["AUC"] if best_row else None,
            "last_val_loss": last_row["val_loss"] if last_row else None,
            "last_F1": last_row["F1"] if last_row else None,
        })

    with open("all_epochs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "epoch", "train_loss", "val_loss", "Se", "Sp", "F1", "ACC", "AUC", "epoch_time_s"
        ])
        writer.writeheader()
        writer.writerows(all_rows)

    with open("run_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "status", "epochs_logged", "early_stop_epoch",
            "reported_best_F1", "best_F1_seen", "best_F1_epoch",
            "best_row_Se", "best_row_Sp", "best_row_ACC", "best_row_AUC",
            "last_val_loss", "last_F1"
        ])
        writer.writeheader()
        writer.writerows(summary)

    print(f"Parsed {len(files)} file(s), {len(all_rows)} total epoch records.")
    print("Wrote: all_epochs.csv, run_summary.csv\n")
    print(f"{'file':<28}{'status':<14}{'epochs':<8}{'best_F1':<10}{'@epoch':<8}{'AUC':<8}")
    for s in summary:
        bf = s["best_F1_seen"]
        auc = s["best_row_AUC"]
        print(f"{s['file']:<28}{s['status']:<14}{s['epochs_logged']:<8}"
              f"{(f'{bf:.4f}' if bf is not None else '-'):<10}"
              f"{str(s['best_F1_epoch']):<8}"
              f"{(f'{auc:.4f}' if auc is not None else '-'):<8}")


if __name__ == "__main__":
    main()
