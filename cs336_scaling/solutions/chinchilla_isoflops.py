import json

import matplotlib.pyplot as plt
import numpy as np


def get_optimal_runs(data):
    budgets = sorted((set(d["compute_budget"] for d in data)))

    optimal_runs = []
    for budget in budgets:
        runs = [d for d in data if d["compute_budget"] == budget]

        best_run = min(runs, key=lambda x: x["final_loss"])

        optimal_runs.append(best_run)

    return optimal_runs


def fit_scaling_law(C, N_opt):
    log_C = np.log(C / 1e18)
    log_N = np.log(N_opt)

    alpha, log_A = np.polyfit(log_C, log_N, 1)

    A = np.exp(log_A)
    return A, alpha


def predict_optimal_parameters(compute_budget, A, alpha):
    return A * (compute_budget / 1e18) ** alpha


def create_scaling_law_plot(C, N_opt, A, alpha):
    N_1e23 = predict_optimal_parameters(1e23, A, alpha)
    N_1e24 = predict_optimal_parameters(1e24, A, alpha)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        C,
        N_opt,
        color="blue",
        s=70,
        label="Observed compute-optimal configurations",
        zorder=3,
    )

    # Fitted region
    C_observed = np.logspace(np.log10(C.min()), np.log10(C.max()), 150)

    ax.plot(
        C_observed,
        predict_optimal_parameters(C_observed, A, alpha),
        color="red",
        linewidth=2,
        label="Fitted scaling law",
    )

    # Extrapolated region
    C_extrapolated = np.logspace(np.log10(C.max()), np.log10(1e24), 150)

    ax.plot(
        C_extrapolated,
        predict_optimal_parameters(C_extrapolated, A, alpha),
        color="red",
        linestyle="--",
        linewidth=2,
        label="Extrapolation",
    )

    # Target compute budgets
    ax.scatter(
        [1e23, 1e24],
        [N_1e23, N_1e24],
        color="green",
        marker="*",
        s=180,
        label="Extrapolated predictions",
        zorder=4,
    )

    ax.axvline(1e23, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(1e24, color="gray", linestyle="--", alpha=0.5)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Compute budget (FLOPs)")
    ax.set_ylabel("Optimal model size (parameters)")
    ax.set_title("Compute-optimal model size scaling law")

    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig("compute_optimal_scaling.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    with open("../../data/isoflops_curves.json", "r") as file:
        data = json.load(file)

    optimal_runs = get_optimal_runs(data)

    C = np.array([d["compute_budget"] for d in optimal_runs])
    N_opt = np.array([d["parameters"] for d in optimal_runs])

    print("Compute budget: ", C)
    print("Optimal parameters: ", N_opt)

    A, alpha = fit_scaling_law(C, N_opt)

    N_1e23 = predict_optimal_parameters(1e23, A, alpha)
    N_1e24 = predict_optimal_parameters(1e24, A, alpha)

    print(f"Optimal parameters at 1e23 FLOPs: {N_1e23:.4e}")
    print(f"Optimal parameters at 1e24 FLOPs: {N_1e24:.4e}")

    create_scaling_law_plot(C, N_opt, A, alpha)
