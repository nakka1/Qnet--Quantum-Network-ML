#!/usr/bin/env python3
"""Figures for the article, drawn from the same CSVs the HTML report reads.

Kept separate from ``qnet_autopilot.validation.realism_report`` on purpose: the
report draws for a screen, in two themes, with a wide layout; a paper figure is
greyscale-safe, 12 cm across and has to stay legible at print size. Sharing the
renderer would mean one of the two ends up compromised.

    python paper/make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "runs"
FIGS = Path(__file__).resolve().parent / "figs"

#: Marker and dash per series, so the figures survive a greyscale print.
STYLES = [("o", "-"), ("s", "--"), ("^", "-."), ("D", ":"), ("v", (0, (3, 1, 1, 1)))]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})


def _save(figure, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(pad=0.4)
    figure.savefig(FIGS / f"{name}.pdf")
    plt.close(figure)
    print(f"  wrote figs/{name}.pdf")


def figure_blocking() -> None:
    """Who chooses the block boundary, as the link gets harder."""
    stress = pd.read_csv(RUNS / "realism/blocking_stress_summary.csv")
    stress = stress.sort_values("vibration_amplitude_rad")
    figure, axes = plt.subplots(figsize=(4.9, 3.1))

    series = [("timer_1s_hz", "temporizador de 1 s"),
              ("best_timer_hz", "melhor temporizador (retrospectiva)"),
              ("adaptive_hz", "controlador contíguo"),
              ("pooling_hz", "controlador agrupado"),
              ("pooled_oracle_hz", "oráculo agrupado")]
    for (column, label), (marker, dash) in zip(series, STYLES):
        axes.plot(stress["vibration_amplitude_rad"],
                  np.maximum(stress[column], 1.0), marker=marker,
                  markersize=4, linewidth=1.3, linestyle=dash, label=label,
                  color="black" if column == "pooled_oracle_hz" else None)

    zero = stress[stress["timer_1s_hz"] <= 0.0]
    if not zero.empty:
        axes.annotate("zero", xy=(float(zero["vibration_amplitude_rad"].iloc[0]), 1.0),
                      xytext=(-2, 6), textcoords="offset points", fontsize=7,
                      ha="right")
    axes.set_yscale("log")
    axes.set_xlabel("amplitude RMS do tom de rede (rad)")
    axes.set_ylabel("chave secreta na execução (bit/s)")
    axes.legend(loc="lower left", frameon=False)
    _save(figure, "blocagem")


def figure_round_cost() -> None:
    """The heralding cost ratio: a constant, not an additive handshake."""
    cost = pd.read_csv(RUNS / "sequence/round_cost.csv").sort_values("span_km")
    figure, (left, right) = plt.subplots(1, 2, figsize=(5.4, 2.4))

    left.plot(cost["span_km"], cost["sequence_ms_per_round"], marker="o",
              markersize=4, linewidth=1.3, label="SeQUeNCe")
    left.plot(cost["span_km"], cost["analytic_ms_per_round"], marker="s",
              markersize=4, linewidth=1.3, linestyle="--", label="analítico")
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("vão (km)")
    left.set_ylabel("ms por tentativa")
    left.legend(loc="upper left", frameon=False)

    right.plot(cost["span_km"], cost["ratio"], marker="^", markersize=4,
               linewidth=1.3, color="black")
    right.axhline(4.1, linestyle=":", linewidth=1.0, color="grey")
    right.set_xscale("log")
    right.set_ylim(0, 5)
    right.set_xlabel("vão (km)")
    right.set_ylabel("razão")
    right.annotate("4,1", xy=(11, 4.25), fontsize=8, color="grey")
    _save(figure, "custo-heralding")


def figure_finite_key() -> None:
    """What the asymptotic formula was overstating, per compensator."""
    link = pd.read_csv(RUNS / "realism/finite_key_link.csv")
    figure, axes = plt.subplots(figsize=(4.9, 2.9))

    order = ["MLP", "ridge (Wiener)", "filtro alfa-beta", "persistência"]
    for name, (marker, dash) in zip(order, STYLES):
        group = link[link["compensator"] == name].sort_values("block_duration_s")
        if group.empty:
            continue
        axes.plot(group["block_duration_s"], group["ratio"], marker=marker,
                  markersize=4, linewidth=1.3, linestyle=dash, label=name)

    axes.set_xscale("log")
    axes.set_xlabel("duração do bloco (s)")
    axes.set_ylabel("chave finita / assintótica")
    axes.set_ylim(0, 1.0)
    axes.legend(loc="upper left", frameon=False)
    _save(figure, "chave-finita")


def figure_attempt_cost() -> None:
    """The prize left for a learner, against refusing to play."""
    floor = pd.read_csv(RUNS / "attempt_cost/floor_summary.csv")
    figure, axes = plt.subplots(figsize=(4.9, 2.7))

    for (scale, group), (marker, dash) in zip(floor.groupby("deadline_scale"),
                                              STYLES):
        ordered = group.sort_values("attempt_cost")
        axes.plot(ordered["attempt_cost"], ordered["over_doing_nothing"],
                  marker=marker, markersize=4, linewidth=1.3, linestyle=dash,
                  label=f"prazo $\\times${scale:g}")

    axes.axvline(0.01, linestyle=":", linewidth=1.0, color="grey")
    axes.annotate("custo publicado", xy=(0.0098, 0.56), fontsize=7,
                  color="grey", ha="right", rotation=90, va="top")
    axes.set_xlabel("custo por tentativa de geração")
    axes.set_ylabel("melhor heurística $-$ não fazer nada")
    axes.legend(loc="upper right", frameon=False)
    _save(figure, "custo-tentativa")


def figure_conditioning() -> None:
    """One model across the band, per held-out tone."""
    detail = pd.read_csv(RUNS / "field/conditioning.csv")
    figure, axes = plt.subplots(figsize=(4.9, 3.0))

    order = ["especialista, mesma frequência", "filtro alfa-beta",
             "janela longa, densa", "janela longa, diluída",
             "janela curta, densa (fase 3)"]
    labels = ["especialista (uma por frequência)", "filtro alfa-beta",
              "generalista, janela longa densa", "generalista, janela diluída",
              "generalista, janela curta"]
    frequencies = sorted(detail["test_frequency_hz"].unique())
    for variant, label, (marker, dash) in zip(order, labels, STYLES):
        group = detail[detail["variant"] == variant]
        if group.empty:
            continue
        means = [float(group[group["test_frequency_hz"] == frequency]
                       ["mean_qber"].mean()) for frequency in frequencies]
        axes.plot(frequencies, means, marker=marker, markersize=4,
                  linewidth=1.3, linestyle=dash, label=label,
                  color="black" if "especialista" in variant else None)

    axes.set_yscale("log")
    axes.set_xlabel("frequência do tom de teste (Hz)")
    axes.set_ylabel("QBER média (3 sementes de ajuste)")
    axes.legend(loc="lower left", frameon=False, ncol=1)
    _save(figure, "condicionamento")


def main() -> int:
    print("drawing article figures")
    for draw in (figure_blocking, figure_round_cost, figure_finite_key,
                 figure_attempt_cost, figure_conditioning):
        try:
            draw()
        except FileNotFoundError as error:
            print(f"  skipped {draw.__name__}: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
