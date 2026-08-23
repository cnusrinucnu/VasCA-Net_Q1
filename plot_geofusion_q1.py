import os
import csv
import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# PATHS
# ==========================================================

HISTORY_PATH = (
    "./results_q1/drive/geofusion/"
    "training_history.csv"
)

TEST_PATH = (
    "./results_q1/drive/geofusion_sliding/"
    "test_metrics.csv"
)

OUTPUT_DIR = (
    "./results_q1/drive/geofusion_plots"
)


# ==========================================================
# STYLE
# ==========================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


# ==========================================================
# UTILITY
# ==========================================================

def save_figure(fig, name):

    png_path = os.path.join(
        OUTPUT_DIR,
        name + ".png"
    )

    pdf_path = os.path.join(
        OUTPUT_DIR,
        name + ".pdf"
    )

    fig.savefig(
        png_path,
        bbox_inches="tight",
        dpi=300,
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    print("Saved:", png_path)
    print("Saved:", pdf_path)

    plt.close(fig)


# ==========================================================
# LOAD CSV
# ==========================================================

def load_csv(path):

    rows = []

    with open(path, "r") as f:

        reader = csv.DictReader(f)

        for row in reader:

            cleaned = {}

            for key, value in row.items():

                try:
                    cleaned[key] = float(value)

                except:
                    cleaned[key] = value

            rows.append(cleaned)

    return rows


# ==========================================================
# FIND COLUMN
# ==========================================================

def find_column(
    columns,
    candidates,
):

    for candidate in candidates:

        for col in columns:

            if candidate.lower() == col.lower():

                return col

    for candidate in candidates:

        for col in columns:

            if candidate.lower() in col.lower():

                return col

    return None


# ==========================================================
# MAIN
# ==========================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    print("=" * 70)
    print("GEOFUSION-VASCA-NET Q1 PLOT GENERATION")
    print("=" * 70)

    # ------------------------------------------------------
    # Load history
    # ------------------------------------------------------

    history = load_csv(
        HISTORY_PATH
    )

    print(
        "Training history epochs:",
        len(history)
    )

    columns = list(
        history[0].keys()
    )

    print(
        "History columns:",
        columns
    )

    epochs = np.arange(
        1,
        len(history) + 1,
    )

    # ------------------------------------------------------
    # Identify columns
    # ------------------------------------------------------

    def get_values(candidates):

        column = find_column(
            columns,
            candidates,
        )

        if column is None:

            print(
                "WARNING: Column not found:",
                candidates
            )

            return None

        return np.array(
            [
                float(row[column])
                for row in history
            ]
        )


    train_loss = get_values([
        "train_loss",
        "Train Loss",
    ])

    val_loss = get_values([
        "val_loss",
        "Val Loss",
    ])

    train_seg = get_values([
        "train_seg",
        "seg_loss",
        "Train Seg",
    ])

    train_geo = get_values([
        "train_geo",
        "geo_loss",
        "Train Geo",
    ])

    val_dice = get_values([
        "val_dice",
        "Dice",
    ])

    val_iou = get_values([
        "val_iou",
        "IoU",
    ])

    val_se = get_values([
        "val_se",
        "Se",
        "Sensitivity",
    ])

    val_sp = get_values([
        "val_sp",
        "Sp",
        "Specificity",
    ])

    val_precision = get_values([
        "val_precision",
        "Precision",
    ])

    val_acc = get_values([
        "val_acc",
        "ACC",
        "Accuracy",
    ])

    lr = get_values([
        "lr",
        "learning_rate",
    ])


    # ======================================================
    # PLOT 1
    # LOSS CURVE
    # ======================================================

    if train_loss is not None:

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.plot(
            epochs,
            train_loss,
            marker="o",
            linewidth=2,
            label="Training Loss",
        )

        if val_loss is not None:

            ax.plot(
                epochs,
                val_loss,
                marker="s",
                linewidth=2,
                label="Validation Loss",
            )

        ax.set_xlabel(
            "Epoch"
        )

        ax.set_ylabel(
            "Loss"
        )

        ax.set_title(
            "Training and Validation Loss"
        )

        ax.grid(
            alpha=0.3
        )

        ax.legend()

        save_figure(
            fig,
            "01_loss_curve"
        )


    # ======================================================
    # PLOT 2
    # DICE CURVE
    # ======================================================

    if val_dice is not None:

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.plot(
            epochs,
            val_dice,
            marker="o",
            linewidth=2,
            label="Validation Dice",
        )

        best_epoch = (
            np.argmax(val_dice)
            +
            1
        )

        best_dice = np.max(
            val_dice
        )

        ax.scatter(
            best_epoch,
            best_dice,
            s=100,
            zorder=5,
            label=(
                f"Best: {best_dice:.4f}"
            ),
        )

        ax.axvline(
            best_epoch,
            linestyle="--",
            alpha=0.6,
        )

        ax.set_xlabel(
            "Epoch"
        )

        ax.set_ylabel(
            "Dice Score"
        )

        ax.set_title(
            "Validation Dice Performance"
        )

        ax.grid(
            alpha=0.3
        )

        ax.legend()

        save_figure(
            fig,
            "02_dice_curve"
        )


    # ======================================================
    # PLOT 3
    # IoU CURVE
    # ======================================================

    if val_iou is not None:

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.plot(
            epochs,
            val_iou,
            marker="o",
            linewidth=2,
        )

        ax.set_xlabel(
            "Epoch"
        )

        ax.set_ylabel(
            "IoU"
        )

        ax.set_title(
            "Validation IoU Performance"
        )

        ax.grid(
            alpha=0.3
        )

        save_figure(
            fig,
            "03_iou_curve"
        )


    # ======================================================
    # PLOT 4
    # SEGMENTATION AND GEOMETRY LOSS
    # ======================================================

    if (
        train_seg is not None
        or train_geo is not None
    ):

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        if train_seg is not None:

            ax.plot(
                epochs,
                train_seg,
                marker="o",
                linewidth=2,
                label="Segmentation Loss",
            )

        if train_geo is not None:

            ax.plot(
                epochs,
                train_geo,
                marker="s",
                linewidth=2,
                label="Geometry Loss",
            )

        ax.set_xlabel(
            "Epoch"
        )

        ax.set_ylabel(
            "Loss"
        )

        ax.set_title(
            "Segmentation and Geometry Loss"
        )

        ax.grid(
            alpha=0.3
        )

        ax.legend()

        save_figure(
            fig,
            "04_segmentation_geometry_loss"
        )


    # ======================================================
    # PLOT 5
    # CLINICAL METRICS
    # ======================================================

    metric_arrays = []

    if val_dice is not None:
        metric_arrays.append(
            ("Dice", val_dice)
        )

    if val_iou is not None:
        metric_arrays.append(
            ("IoU", val_iou)
        )

    if val_se is not None:
        metric_arrays.append(
            ("Sensitivity", val_se)
        )

    if val_sp is not None:
        metric_arrays.append(
            ("Specificity", val_sp)
        )

    if val_precision is not None:
        metric_arrays.append(
            ("Precision", val_precision)
        )

    if val_acc is not None:
        metric_arrays.append(
            ("Accuracy", val_acc)
        )


    if len(metric_arrays) > 0:

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        for name, values in metric_arrays:

            ax.plot(
                epochs,
                values,
                marker="o",
                linewidth=2,
                label=name,
            )

        ax.set_xlabel(
            "Epoch"
        )

        ax.set_ylabel(
            "Score"
        )

        ax.set_title(
            "Validation Performance Metrics"
        )

        ax.grid(
            alpha=0.3
        )

        ax.legend(
            ncol=3
        )

        save_figure(
            fig,
            "05_validation_metrics"
        )


    # ======================================================
    # PLOT 6
    # LEARNING RATE
    # ======================================================

    if lr is not None:

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.plot(
            epochs,
            lr,
            marker="o",
            linewidth=2,
        )

        ax.set_xlabel(
            "Epoch"
        )

        ax.set_ylabel(
            "Learning Rate"
        )

        ax.set_title(
            "Learning Rate Schedule"
        )

        ax.set_yscale(
            "log"
        )

        ax.grid(
            alpha=0.3
        )

        save_figure(
            fig,
            "06_learning_rate"
        )


    # ======================================================
    # PLOT 7
    # Q1 TRAINING DASHBOARD
    # ======================================================

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
    )

    # Loss

    if train_loss is not None:

        axes[0, 0].plot(
            epochs,
            train_loss,
            marker="o",
            label="Train",
        )

    if val_loss is not None:

        axes[0, 0].plot(
            epochs,
            val_loss,
            marker="s",
            label="Validation",
        )

    axes[0, 0].set_title(
        "Training and Validation Loss"
    )

    axes[0, 0].set_xlabel(
        "Epoch"
    )

    axes[0, 0].set_ylabel(
        "Loss"
    )

    axes[0, 0].legend()

    axes[0, 0].grid(
        alpha=0.3
    )


    # Dice and IoU

    if val_dice is not None:

        axes[0, 1].plot(
            epochs,
            val_dice,
            marker="o",
            label="Dice",
        )

    if val_iou is not None:

        axes[0, 1].plot(
            epochs,
            val_iou,
            marker="s",
            label="IoU",
        )

    axes[0, 1].set_title(
        "Segmentation Performance"
    )

    axes[0, 1].set_xlabel(
        "Epoch"
    )

    axes[0, 1].set_ylabel(
        "Score"
    )

    axes[0, 1].legend()

    axes[0, 1].grid(
        alpha=0.3
    )


    # Se / Precision

    if val_se is not None:

        axes[1, 0].plot(
            epochs,
            val_se,
            marker="o",
            label="Sensitivity",
        )

    if val_precision is not None:

        axes[1, 0].plot(
            epochs,
            val_precision,
            marker="s",
            label="Precision",
        )

    axes[1, 0].set_title(
        "Sensitivity and Precision"
    )

    axes[1, 0].set_xlabel(
        "Epoch"
    )

    axes[1, 0].set_ylabel(
        "Score"
    )

    axes[1, 0].legend()

    axes[1, 0].grid(
        alpha=0.3
    )


    # Accuracy / Specificity

    if val_sp is not None:

        axes[1, 1].plot(
            epochs,
            val_sp,
            marker="o",
            label="Specificity",
        )

    if val_acc is not None:

        axes[1, 1].plot(
            epochs,
            val_acc,
            marker="s",
            label="Accuracy",
        )

    axes[1, 1].set_title(
        "Specificity and Accuracy"
    )

    axes[1, 1].set_xlabel(
        "Epoch"
    )

    axes[1, 1].set_ylabel(
        "Score"
    )

    axes[1, 1].legend()

    axes[1, 1].grid(
        alpha=0.3
    )

    fig.suptitle(
        "GeoFusion-VasCA-Net: Training Dynamics",
        fontsize=18,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "07_q1_training_dashboard"
    )


    # ======================================================
    # LOAD TEST RESULTS
    # ======================================================

    if os.path.exists(
        TEST_PATH
    ):

        test_rows = load_csv(
            TEST_PATH
        )

        print(
            "Test samples:",
            len(test_rows)
        )

        if len(test_rows) > 0:

            test_columns = list(
                test_rows[0].keys()
            )

            print(
                "Test columns:",
                test_columns
            )


            # --------------------------------------------------
            # Numeric metrics
            # --------------------------------------------------

            numeric_columns = []

            for col in test_columns:

                try:

                    values = [
                        float(
                            row[col]
                        )
                        for row in test_rows
                    ]

                    numeric_columns.append(
                        col
                    )

                except:
                    pass


            # --------------------------------------------------
            # Per-image Dice
            # --------------------------------------------------

            dice_col = find_column(
                test_columns,
                ["Dice"]
            )

            iou_col = find_column(
                test_columns,
                ["IoU"]
            )

            se_col = find_column(
                test_columns,
                ["Se", "Sensitivity"]
            )

            sp_col = find_column(
                test_columns,
                ["Sp", "Specificity"]
            )

            precision_col = find_column(
                test_columns,
                ["Precision"]
            )

            acc_col = find_column(
                test_columns,
                ["ACC", "Accuracy"]
            )


            labels = [
                f"Image {i+1}"
                for i in range(
                    len(test_rows)
                )
            ]


            if dice_col is not None:

                dice_values = np.array(
                    [
                        float(
                            row[dice_col]
                        )
                        for row in test_rows
                    ]
                )

                fig, ax = plt.subplots(
                    figsize=(8, 5)
                )

                bars = ax.bar(
                    labels,
                    dice_values,
                )

                ax.axhline(
                    dice_values.mean(),
                    linestyle="--",
                    linewidth=2,
                    label=(
                        f"Mean = "
                        f"{dice_values.mean():.4f}"
                    ),
                )

                for bar, value in zip(
                    bars,
                    dice_values,
                ):

                    ax.text(
                        bar.get_x()
                        +
                        bar.get_width() / 2,
                        value + 0.01,
                        f"{value:.4f}",
                        ha="center",
                        va="bottom",
                    )

                ax.set_ylim(
                    0,
                    1
                )

                ax.set_ylabel(
                    "Dice Score"
                )

                ax.set_title(
                    "Per-Image DRIVE Test Dice"
                )

                ax.legend()

                ax.grid(
                    axis="y",
                    alpha=0.3,
                )

                save_figure(
                    fig,
                    "08_test_dice_per_image"
                )


            # --------------------------------------------------
            # Per-image metrics
            # --------------------------------------------------

            available = []

            for name, col in [
                ("Dice", dice_col),
                ("IoU", iou_col),
                ("Se", se_col),
                ("Sp", sp_col),
                ("Precision", precision_col),
                ("ACC", acc_col),
            ]:

                if col is not None:

                    values = [
                        float(row[col])
                        for row in test_rows
                    ]

                    available.append(
                        (
                            name,
                            values,
                        )
                    )


            if len(available) > 0:

                fig, ax = plt.subplots(
                    figsize=(12, 6)
                )

                x = np.arange(
                    len(labels)
                )

                width = (
                    0.8
                    /
                    len(available)
                )

                start = (
                    -
                    (
                        len(available) - 1
                    )
                    *
                    width
                    /
                    2
                )

                for idx, (
                    name,
                    values,
                ) in enumerate(
                    available
                ):

                    ax.bar(
                        x
                        +
                        start
                        +
                        idx
                        *
                        width,
                        values,
                        width,
                        label=name,
                    )

                ax.set_xticks(
                    x
                )

                ax.set_xticklabels(
                    labels
                )

                ax.set_ylim(
                    0,
                    1
                )

                ax.set_ylabel(
                    "Score"
                )

                ax.set_title(
                    "Per-Image DRIVE Test Metrics"
                )

                ax.legend(
                    ncol=3
                )

                ax.grid(
                    axis="y",
                    alpha=0.3,
                )

                save_figure(
                    fig,
                    "09_test_metrics_per_image"
                )


            # --------------------------------------------------
            # Heatmap
            # --------------------------------------------------

            if len(available) > 0:

                metric_names = [
                    item[0]
                    for item in available
                ]

                metric_matrix = np.array(
                    [
                        item[1]
                        for item in available
                    ]
                )

                fig, ax = plt.subplots(
                    figsize=(10, 6)
                )

                im = ax.imshow(
                    metric_matrix,
                    aspect="auto",
                )

                ax.set_xticks(
                    np.arange(
                        len(labels)
                    )
                )

                ax.set_xticklabels(
                    labels
                )

                ax.set_yticks(
                    np.arange(
                        len(metric_names)
                    )
                )

                ax.set_yticklabels(
                    metric_names
                )

                for i in range(
                    metric_matrix.shape[0]
                ):

                    for j in range(
                        metric_matrix.shape[1]
                    ):

                        ax.text(
                            j,
                            i,
                            f"{metric_matrix[i, j]:.3f}",
                            ha="center",
                            va="center",
                            fontsize=10,
                        )

                fig.colorbar(
                    im,
                    ax=ax,
                    label="Score",
                )

                ax.set_title(
                    "GeoFusion-VasCA-Net Test Performance Heatmap"
                )

                fig.tight_layout()

                save_figure(
                    fig,
                    "10_test_metric_heatmap"
                )


            # --------------------------------------------------
            # Radar chart
            # --------------------------------------------------

            if len(available) > 0:

                means = np.array(
                    [
                        np.mean(
                            item[1]
                        )
                        for item in available
                    ]
                )

                names = [
                    item[0]
                    for item in available
                ]

                angles = np.linspace(
                    0,
                    2 * np.pi,
                    len(names),
                    endpoint=False,
                ).tolist()

                values = means.tolist()

                angles += angles[:1]

                values += values[:1]

                fig, ax = plt.subplots(
                    figsize=(7, 7),
                    subplot_kw={
                        "projection": "polar"
                    },
                )

                ax.plot(
                    angles,
                    values,
                    linewidth=2,
                    marker="o",
                )

                ax.fill(
                    angles,
                    values,
                    alpha=0.25,
                )

                ax.set_xticks(
                    angles[:-1]
                )

                ax.set_xticklabels(
                    names
                )

                ax.set_ylim(
                    0,
                    1
                )

                ax.set_title(
                    "GeoFusion-VasCA-Net Test Profile",
                    pad=25,
                )

                save_figure(
                    fig,
                    "11_test_radar"
                )


    print()
    print("=" * 70)
    print("ALL PLOTS GENERATED SUCCESSFULLY")
    print("=" * 70)

    print(
        "Output directory:",
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
