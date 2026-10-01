"""Render Task 06 manuscripts only from complete, verified result artifacts.

Importing this module writes nothing. ``--check`` validates evidence and renders
in memory; the default command writes only the explicitly requested manuscripts,
conditional Task 07 specification, and GitHub description. It never fits,
forecasts, opens a cohort, calls an LLM, or publishes to a remote service.
"""

import argparse
import json
from pathlib import Path
import re

import numpy as np

from .behavior_data import ROOT
from .institutional_data import _verify_frozen_artifacts
from .sources import sha256


QUESTION = "Does a behavioral model fitted to human play predict which allocation rule works, including a rule it never saw?"
PAPER = "https://www.nature.com/articles/s41467-025-58043-7"
DESCRIPTION = "Can learned models of human behavior predict which institution works? Closed-loop validity tests on human common-pool data; evolutionary search planned."
RULES = ("Equal", "Mixed", "Proportional")
SEEDS = (17, 29, 43)
FIGURES = ("task06-institution-benchmark", "task06-attenuation",
           "task06-tipping-sensitivity", "task06-transfer-effects")


def number(value, digits=3):
    if value is None or not np.isfinite(value):
        raise ValueError("A displayed empirical number is missing or nonfinite")
    return f"{float(value):.{digits}f}"


def bounds(values, digits=3):
    if len(values) != 2:
        raise ValueError("Expected a two-sided interval")
    return f"[{number(values[0], digits)}, {number(values[1], digits)}]"


def label(family):
    names = {"T03-constant": "Task 03 constant", "T03-GRU": "Task 03 GRU", "T04-GRU": "Continued GRU"}
    if family.startswith("CL-"):
        return "CL(" + label(family[3:]) + ")"
    return names.get(family, family)


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", r"\|")

    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |", "| " + " | ".join(":---" for _ in headers) + " |",
                      *["| " + " | ".join(map(cell, row)) + " |" for row in rows]])


def _words(text):
    return len(re.findall(r"\S+", text))


def _prose_words(text):
    lines = [line for line in text.splitlines() if not line.lstrip().startswith(("|", "![", "<", "#"))]
    return _words("\n".join(lines))


def _audit_fraction_match(audit):
    fractions = audit["fraction_audit"]
    identity = {"aggregation": "ratio_of_sums_then_groups", "denominator": "actual_offer"}
    if identity not in fractions["matching_variants"]:
        raise RuntimeError("Report refused; the external allocation-weighted fraction match is absent")
    matches = [row for row in fractions["comparisons"] if all(row[key] == value for key, value in identity.items())]
    if len(matches) != 1 or not matches[0]["matches_all_six_at_quoted_three_decimals"]:
        raise RuntimeError("Report refused; the external fraction re-verification is inconsistent")
    return matches[0]


def _audit_paragraph(data):
    audit = data["external_audit"]
    matched = _audit_fraction_match(audit)
    actual = matched["actual"]
    predicted = "/".join(number(actual["predicted_fraction"][rule]) for rule in RULES)
    observed = "/".join(number(actual["observed_fraction"][rule]) for rule in RULES)
    continued = "/".join(number(matched["continued_mixed_per_seed"][str(seed)]) for seed in SEEDS)
    increments = "/".join(number(matched["floor_tilt_predicted_fraction_change"][rule]) for rule in RULES)
    mixed = {row["tau"]: row for row in data["audit_rollouts"]["cells"] if row["rule"] == "Mixed"}
    counts = {tau: sum(int(game["survival"]) for game in row["games"]) for tau, row in mixed.items()}
    return (f"The external audit's six fractions reproduce at quoted precision: predicted E/M/P {predicted}, observed {observed}. "
            "They use nonforced Σc/Σe within each group, then equal groups: allocation-weighted fractions, distinct from Task 06's unchanged choice means. "
            f"Continued Mixed fractions are {continued} (seeds 17/29/43). The historical exp(0.6c/floor(e)) tilt raises E/M/P fractions by {increments}, "
            f"costing {number(audit['floor_tilt_nll_cost_equal_rules'], 4)} NLL nats; Task 06 instead tilts by actual c/e. "
            f"Continuation changes NLL by {number(audit['continuation_nll_change_equal_rules'], 4)} (three-seed mean versus original seed 17). "
            f"Our new Mixed audit draws survive {counts[0.]}/{mixed[0.]['count']}→{counts[.6]}/{mixed[.6]['count']} "
            f"(MC SE {number(mixed[0.]['survival_mcse'])}/{number(mixed[.6]['survival_mcse'])}). "
            "The external 64-game seed was unavailable, preventing exact bank replication. "
            "[Audit artifacts](../results/task06/external_audit_predictions.json) retain all rechecks and estimand definitions.")


def evidence(root=ROOT):
    """Fail closed when any required run, forecast, inference or figure is absent."""
    root = Path(root)
    folder = root / "results/task06"
    required = {
        "config": root / "configs/task06.json",
        **{key: folder / name for key, name in {
            "training": "training_complete.json", "transfer": "transfer.json",
            "transfer_nll": "diagnostics_experiment2.json", "figures": "figures.json",
            "freeze": "freeze_pushed.json", "manifest": "manifest.json",
            "benchmark": "benchmark.json", "decision": "headroom_decision.json",
            "diagnostics": "diagnostics_validation.json", "nll_cost": "calibration_nll_cost.json",
            "calibration": "calibration.json", "integration": "integration_checks.json",
            "tilted_integration": "tilted_integration_verification.json",
            "fresh_process": "fresh_process_verification.json",
            "preservation": "preservation_verification.json",
            "source_audit": "institutional_source_audit.json",
            "external_audit": "external_audit_predictions.json",
            "audit_rollouts": "external_audit_rollouts.json",
            "feature_audit": "feature_development_audit.json",
            "recorded": "recorded_games.json", "replay": "recorded_interpolating_replay.json",
            "transfer_replay": "experiment2_replay.json",
            "runtime": "runtime.json",
        }.items()},
    }
    missing = [str(path.relative_to(root)) for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError("Report refused; incomplete Task06 evidence: " + ", ".join(missing))
    data = {key: json.loads(path.read_text()) for key, path in required.items()}
    config, training, transfer = data["config"], data["training"], data["transfer"]
    families = config["reference_families"] + config["fa_families"] + ["CL-" + f for f in config["calibration_families"]]
    best = training["fa_best"]
    if training["fits"] != 9 or training["epochs_per_fit"] != 480 or best not in config["fa_families"]:
        raise RuntimeError("Report refused; all nine fixed-budget FA fits are required")
    feature_audit = data["feature_audit"]
    if (feature_audit["groups"] != 128 or feature_audit["rounds"] != 5120
            or set(feature_audit["by_rule"]) != set((*RULES, "M1"))
            or any(r["rounds"] != 1280 or not 0 <= r["clipped"] <= r["rounds"]
                   for r in feature_audit["by_rule"].values())
            or feature_audit["clipped_rounds"] != sum(r["clipped"] for r in feature_audit["by_rule"].values())):
        raise RuntimeError("Report refused; feature clipping audit coverage or counts are inconsistent")
    if (data["transfer_nll"]["checkpoints"] != 57 or data["transfer_nll"]["human_groups"] != 120
            or data["diagnostics"]["checkpoints"] != 57 or data["diagnostics"]["human_groups"] != 32):
        raise RuntimeError("Report refused; transfer/validation likelihood coverage is incomplete")
    expected_nll = {(family, rule) for family in families for rule in ("Proportional", "Interpolating", "M1")}
    nll_rows = data["transfer_nll"]["per_rule"]
    if ({(r["family"], r["mechanism"]) for r in nll_rows} != expected_nll
            or len(nll_rows) != len(expected_nll) or any(r["groups"] != 40 or r["seeds"] != 3 for r in nll_rows)):
        raise RuntimeError("Report refused; every learned line requires all three 40-group transfer conditions")
    transfer_replay = data["transfer_replay"]["summaries"]
    if (len(transfer_replay) != 2
            or {(r["cohort"], r["mechanism"]) for r in transfer_replay}
            != {("Exp2", rule) for rule in ("Proportional", "Interpolating")}
            or any(r["rows"] != 1600 or not np.isfinite(r["max_abs_offer_residual"])
                   or r["max_abs_offer_residual"] < 0 for r in transfer_replay)):
        raise RuntimeError("Report refused; complete finite human transfer allocation replay is required")
    if (transfer["fa_best"] != best or transfer["primary_simulators"] != ["BC1", best, "CL-" + best]
            or data["decision"]["status"] != "complete" or transfer["D1"] != data["decision"]["outcome"]):
        raise RuntimeError("Report refused; selected family, D1 or primary comparison identities disagree")
    if (data["freeze"]["manifest_sha256"] != sha256(required["manifest"])
            or data["freeze"]["commit"] != transfer["freeze_commit"]
            or data["freeze"]["analysis_code_sha256"] != data["manifest"]["analysis_code_sha256"]):
        raise RuntimeError("Report refused; scoring does not identify the pushed forecast freeze")
    for name, digest in data["manifest"]["analysis_code_sha256"].items():
        if not (root / name).is_file() or sha256(root / name) != digest:
            raise RuntimeError(f"Report refused; frozen analysis changed: {name}")
    _verify_frozen_artifacts(root, data["manifest"])
    for key in ("integration", "tilted_integration", "fresh_process", "preservation"):
        if not data[key]["passed"]:
            raise RuntimeError(f"Report refused; numerical/preservation gate did not pass: {key}")
    if not data["calibration"]["complete"] or len(data["calibration"]["selected"]) != 18:
        raise RuntimeError("Report refused; all eighteen calibrations are required")
    if len(data["nll_cost"]["per_family_all_rules"]) != 6:
        raise RuntimeError("Report refused; six validation calibration costs are required")
    audit_expected = {(family, seed, rule) for family, seeds in
                      (("T03-GRU", (17,)), ("Audit-T03-GRU-floor-tilt-0.6", (17,)), ("T04-GRU", SEEDS))
                      for seed in seeds for rule in RULES}
    if {(r["family"], r["seed"], r["mechanism"]) for r in data["external_audit"]["per_seed_rule"]} != audit_expected:
        raise RuntimeError("Report refused; external teacher-forced audit coverage is incomplete")
    _audit_fraction_match(data["external_audit"])
    audit_cells = data["audit_rollouts"]["cells"]
    if (len(audit_cells) != 6 or {(r["rule"], r["tau"]) for r in audit_cells} != {(r, t) for r in RULES for t in (0., .6)}
            or any(r["count"] != 64 or len(r["games"]) != 64 for r in audit_cells)):
        raise RuntimeError("Report refused; external audit requires all six declared 64-game cells")
    for name in FIGURES:
        for suffix in ("png", "svg"):
            relative = f"docs/assets/{name}.{suffix}"
            path = root / relative
            if not path.is_file() or data["figures"]["figures_sha256"].get(relative) != sha256(path):
                raise RuntimeError(f"Report refused; required figure missing or changed: {relative}")
    for name, digest in data["figures"]["inputs_sha256"].items():
        if sha256(root / name) != digest:
            raise RuntimeError(f"Report refused; a plotted input changed: {name}")
    tests = data["runtime"].get("invocations", {}).get("test", [])
    if not tests or tests[-1]["exit_code"] != 0:
        raise RuntimeError("Report refused; full existing test suite has no successful latest receipt")
    data["fits"] = []
    for family in config["fa_families"]:
        for seed in SEEDS:
            path = folder / "weights" / f"{family}_{seed}" / "metadata.json"
            fitted = json.loads(path.read_text())
            if fitted["epochs"] != 480:
                raise RuntimeError("Report refused; a FA fit is incomplete")
            data["fits"].append(fitted)
    data["sensitivity"] = {}
    for family in families:
        for seed in SEEDS:
            for tau in config["sensitivity"]["taus"]:
                cell = json.loads((folder / "sensitivity" / f"{family}_{seed}_{tau:+.1f}.json").read_text())
                record = cell["contract"]["record"]
                if (record["family"], record["seed"], cell["contract"]["tau"]) != (family, seed, tau):
                    raise RuntimeError("Report refused; sensitivity-cell identity differs")
                if any(cell["cells"][rule]["simulation"]["games"] != 256 for rule in RULES):
                    raise RuntimeError("Report refused; sensitivity sampling budget is incomplete")
                data["sensitivity"][family, seed, tau] = cell["cells"]
    return data


def _tables(data):
    best, transfer = data["training"]["fa_best"], data["transfer"]
    selected = ("BC1", best, "CL-" + best, "T03-GRU", "T03-constant")
    benchmark = {(r["family"], r["rule"]): r for r in data["benchmark"]["populations"]["all"]["rows"]}
    human = [number(benchmark["BC1", rule]["observed"]["surplus"]["mean"]) for rule in RULES]
    rows = [["Humans (40 groups/rule)", *human]]
    for family in ("BC1", "T03-constant", "T03-GRU", best, "CL-" + best):
        n = "512 recorded games/rule" if family == "BC1" else "1,536 games/rule"
        rows.append([label(family) + f" ({n})", *[number(benchmark[family, rule]["predicted"]["surplus"]["mean"]) for rule in RULES]])
    output = {"benchmark": table(["Mean surplus/player/round", *RULES], rows)}
    output["fits"] = table(["FA family", "Mean validation NLL", "Selected epochs (17/29/43)"],
                            [[family, number(data["training"]["mean_validation_nll"][family], 4),
                              "/".join(str(next(r["selected_epoch"] for r in data["fits"] if r["family"] == family and r["seed"] == seed)) for seed in SEEDS)]
                             for family in data["config"]["fa_families"]])
    output["calibration"] = table(["Calibrated line", "Validation NLL cost (nats)"],
                                   [[label(r["calibrated_family"]), number(r["cost"], 4)]
                                    for r in data["nll_cost"]["per_family_all_rules"]])
    diagnostics = {row["family"]: row for row in data["diagnostics"]["attenuation"]}
    rules = {(r["family"], r["mechanism"]): r for r in data["diagnostics"]["per_rule"]}
    output["attenuation"] = table(["Simulator", "Predicted return fraction E/M/P", "Attenuation index"],
        [[label(family), "/".join(number(rules[family, rule]["predicted_fraction"]) for rule in RULES),
          number(diagnostics[family]["attenuation_index"])] for family in ("T03-constant", "T03-GRU", "T04-GRU", best, "CL-" + best)])
    sensitivity = []
    for family in ("T03-GRU", best, "CL-" + best):
        values = {}
        for tau in (0., .6):
            cells = [data["sensitivity"][family, seed, tau]["Mixed"] for seed in SEEDS]
            values[tau] = {metric: float(np.mean([c["simulation"][metric] for c in cells])) for metric in ("surplus", "survival")}
            values[tau]["nll"] = float(np.mean([c["validation"]["nll"] for c in cells]))
        sensitivity.append([label(family), number(values[.6]["nll"] - values[0.]["nll"], 4),
                            f"{number(values[0.]['survival'])} → {number(values[.6]['survival'])}",
                            f"{number(values[0.]['surplus'])} → {number(values[.6]['surplus'])}"])
    output["sensitivity"] = table(["Mixed; additional τ=0→0.6", "Validation ΔNLL", "Survival", "Surplus"], sensitivity)
    output["D1"] = table(["Candidate", "Groups/rule", "IE [95% interval]", "Matched BC1 IE [95% interval]", "Qualifies"],
        [[label(r["family"]), r["human_groups_per_rule"],
          number(r["comparisons"]["surplus"]["candidate_ie"]["estimate"]) + " " + bounds(r["comparisons"]["surplus"]["candidate_ie"]["ci95"]),
          number(r["comparisons"]["surplus"]["bc1_ie"]["estimate"]) + " " + bounds(r["comparisons"]["surplus"]["bc1_ie"]["ci95"]),
          "yes" if r["qualifies_no_headroom"] else "no"] for r in data["decision"]["candidates"]])
    transfer_rows = []
    for family in selected:
        e1, e2 = (transfer["scores"][family]["surplus"][key] for key in ("E1", "E2"))
        transfer_rows.append([label(family), number(e1["predicted"]), number(e1["absolute_error"]),
                              bounds(e1["paired_difference_ci95"]) + "; " + e1["classification"],
                              number(e2["absolute_error"]), bounds(e2["paired_difference_ci95"]) + "; " + e2["classification"]])
    output["transfer"] = table(["Simulator", "Predicted I−P", "E1 |error|", "E1 Δ|error| vs BC1: 95% CI; D2", "E2 |error|", "E2 Δ|error| vs BC1: 95% CI; D2"], transfer_rows)
    likelihood = {(r["family"], r["mechanism"]): r["nll"] for r in data["transfer_nll"]["per_rule"]}
    output["nll"] = table(["Learned simulator", "Proportional NLL", "Interpolating NLL", "M1 NLL", "All 120 groups"],
        [[label(family), *[number(likelihood[family, rule], 4) for rule in ("Proportional", "Interpolating", "M1")],
          number(np.mean([likelihood[family, rule] for rule in ("Proportional", "Interpolating", "M1")]), 4)]
         for family in (best, "CL-" + best, "T03-GRU", "T03-constant")])
    output["secondary"] = table(["Primary simulator", "Survival E1 |error|", "Survival E2 |error|", "Proportional surplus |error|", "Interpolating energy U", "Interpolating pool20 |error|"],
        [[label(f), number(transfer["scores"][f]["survival"]["E1"]["absolute_error"]),
          number(transfer["scores"][f]["survival"]["E2"]["absolute_error"]),
          number(transfer["scores"][f]["per_rule"]["Proportional"]["errors"]["surplus"]["absolute"]),
          number(transfer["scores"][f]["per_rule"]["Interpolating"]["energy_u"]),
          number(transfer["scores"][f]["per_rule"]["Interpolating"]["errors"]["pool20"]["absolute"])]
         for f in transfer["primary_simulators"]])
    return output


def _recommendation(data):
    transfer = data["transfer"]
    if transfer["D1"] == "headroom":
        return ("Headroom remains under the declared decision rule. Task 07 specifies LLM-guided search over behavioral programs, with a random-search control at matched fitting and simulation budgets and a frozen Experiment 3 test. It is specified, not run. Search must improve agreement with human outcome distributions, never maximize simulated cooperation or design new allocation rules.",
                "Task 07: behavioral-program search, matched-budget random search and fixed-model controls; freeze before opening Experiment 3. No search has run.")
    worse = any(v == "worse" for family, values in transfer["D2"].items() if family != "BC1" for v in values.values())
    if worse:
        return ("The declared rule identifies a transfer failure: development performance left no headroom, but a primary transfer comparison is worse than BC1. Investigate instruction and cohort shift before adding model complexity. Do not run evolutionary search.",
                "Investigate instruction and cohort shift; no evolutionary search is recommended by these results.")
    return ("The contribution is the simple ingredient: development performance meets the incumbent criterion and neither primary candidate is worse than BC1 on the declared transfer comparisons. Recommend no evolutionary search. A paper can proceed from the institution-prediction question, through the BC1 benchmark and tipping-point diagnosis, to the feature/calibration ablations and frozen transfer evaluation.",
            "Develop the simple-ingredient paper: question, incumbent benchmark, tipping-point diagnosis, ablations and frozen transfer. No evolutionary search is recommended.")


def _limitations(data):
    replay = next(r for r in data["replay"]["summaries"] if r["cohort"] == "BC1" and r["mechanism"] == "Interpolating")
    transfer_replay = next(r for r in data["transfer_replay"]["summaries"]
                           if r["cohort"] == "Exp2" and r["mechanism"] == "Interpolating")
    human_residual = float(transfer_replay["max_abs_offer_residual"])
    convention = ("Human transfer replay agrees within 1e−4; the large discrepancy in this comparison concerns BC1. "
                  if human_residual <= 1e-4 else
                  "Human transfer also contains an allocation-convention discrepancy. ")
    return ("Experiment 2 changes participants and provides rule instructions absent in Experiment 1; the models have no instruction input. The within-cohort effect removes shared shifts only to first order. "
            f"Published continuous Interpolating allocation has maximum offer replay residuals {number(replay['max_abs_offer_residual'], 4)} for BC1 and {human_residual:.6g} for human Experiment 2. "
            + convention + "Scores evaluate the prespecified continuous implementation, without claiming exact institutional reproduction. "
            "BC1/BC2 terminal pools are equation-inferred because final next-pool fields are missing. BC1 used a different 537-game training collection, continuous actions and outcome-selected checkpoints; its Interpolating rule was also optimized against that simulator. "
            "The paper's component training counts conflict with its stated total; both are retained in the source audit. Participant identities cannot establish independence across launch groups. Exp 1 test outcomes were already opened, so later diagnostics are not pristine confirmation. "
            "Intervals spanning zero mean not distinguishable, not equivalence. These limits constrain causal and general claims beyond this game.")


def _task07(data):
    return f"""# Task 07 — Evolved behavioral programs

Status: specified, not run. Trigger: Task 06 D1 = **{data['transfer']['D1']}**. Experiment 3 remains closed. This specification authorizes no execution in Task 06.

## Question and target

Can LLM-guided search over interpretable behavioral-model programs improve closed-loop fidelity to human groups beyond the frozen Task 06 models and random program search at a matched budget? Allocation rules remain fixed. Fitness measures agreement with human evidence; higher simulated surplus is never the optimization target. Report institutional performance separately from behavioral validity.

## Evidence and comparisons

Keep the existing Experiment 1 group split and participant-linkage caveat. Fit parameters on the 96 training groups only. Use the 32 validation groups for adaptive program selection. The previously opened 32 test groups and Experiment 2 are descriptive diagnostics, excluded from search prompts, fitness, parameter fitting and selection. Never open Experiment 3 until the final public freeze has been pushed.

Resolve the Interpolating source convention before search and before any Experiment 3 access. The published continuous rule does not reproduce recorded BC1 allocations; synthetic evidence suggests a mixture weight quantized near steps of 0.1, without establishing the complete original implementation. Inspect upstream documentation and replay allowed synthetic/development evidence to specify the future executable convention, numerical tolerances and remaining discrepancies. Obtain source clarification if exact reproduction is needed. Freeze that choice and attribution before search; never tune it against Experiment 3. Task 06 scores remain comparisons under its prespecified continuous implementation, not claims of exact institutional reproduction.

Run three paired search seeds. Each LLM-guided run and its random-search counterpart receives the same allowed program language, proposal count, fitting steps, simulation draws, numerical-validation budget and CPU ceiling. Use a fixed pretrained model/proposal-prompt version for the LLM arm. Random search samples the identical program grammar without LLM ranking or feedback. Include unchanged Task 06 FA-best ({data['training']['fa_best']}), its CL variant, the constant model and GRU as fixed-model controls. Record invalid proposals, duplicates, rejected programs, tokens, wall time and all consumed budgets; failures count against both arms.

Before any proposal or LLM call, profile RAM and measured interpreter/fitting/rollout throughput; freeze exact proposal counts, training starts, optimizer steps and rollout counts in configs/task07.json. Choose a budget that fits the measured machine, apply it equally, and publish it before search. A projected phase above eight hours stops with resumable state; it does not silently shrink the experiment. This resource gate is an explicit unexecuted prerequisite, not a completed budget measurement.

## Program space and search

Programs map current public offers, current pool and past-only resident history to a normalized PMF on integers 0..floor(offer). Permitted ingredients include bounded interpretable state updates, persistence, nonlinear responses to public institutional signals and mixtures of valid emissions. Every resident owns separate state; all choices are simultaneous. A persistent random effect, if present, is sampled once per resident. Prohibit rule labels, scheduled remaining rounds, current/future human actions, future offers, outcome lookup tables, data access and unrestricted code execution. M1 and M2 remain nonexecutable.

Run untrusted programs in a restricted interpreter or sandbox with explicit time/memory limits and an allowlist of operations. Reject leakage, unsupported mass, invalid accounting, nonfinite gradients and uncheckpointable state before scoring. Preserve proposal text and source hashes. Evaluate all proposals under the same training and validation schedule and indexed common random numbers; do not give the LLM arm extra rescue fits.

## Fitness and selection

Primary development fitness is the equal-rule sum of two-sample off-diagonal energy U statistics on (mean surplus/10, pool20/200, pool40/200), under Equal, Mixed and Proportional. Fit ordinary response parameters against training human choices; any closed-loop calibration uses training outcomes only. Candidate selection uses validation fidelity, with validation choice NLL as a declared secondary measure and deterministic tie-break. Keep survival, inequality, exclusion and calibration diagnostics separate. Report the observed validation survival tie without replacing it by a preferred institutional order.

Compare best validated programs across paired search seeds, retaining full search curves and all attempted candidates. Report program complexity, parameter counts and compute. Attribute improvements to search only if they exceed both matched-budget random search and frozen controls on final evidence; a higher training score alone is insufficient.

## Public freeze and Experiment 3

Before touching Experiment 3 rows, freeze selected source programs, learned parameters, any tilts, checkpoints, all rollout seeds, complete forecasts, primary estimands, bootstrap rules and analysis-code hashes; commit and push the manifest. Use the inventory's complete-group keys and verify cohort disjointness on opening. Experiment 3 provides Interpolating and recorded M2 groups: score free Interpolating forecasts against its 80 groups; use M2 offers for teacher-forced likelihood only. Do not reconstruct M2 or claim an executable Interpolating–M2 counterfactual contrast.

Primary final comparison is the paired difference in Interpolating distributional fidelity between each search arm and the fixed incumbent, using the same human-group bootstrap samples and separate Monte Carlo uncertainty. Specify the precise incumbent, effect size, multiplicity treatment and decision rule in the public configuration before search. Report 40-round surplus, survival, pool trajectories, inequality and choice likelihood, with all seeds and negative results. Do not adapt programs or selection after opening.

## Completion and interpretation

Verify legal support, information timing, independent resident states, deterministic regeneration, energy-estimator correctness, bootstrap pairing and protected Task 01–06 artifacts. Use one compute thread, sequential resumable jobs and consolidated phase resource receipts. Publish results and interpretable programs without changing the allocation mechanisms. No claims of evolutionary improvement, learning during deployment or recursive self-improvement follow without corresponding evidence. A separate LLM-agent baseline may be future work; this task searches behavioral programs.
"""


def render(data, root=ROOT):
    root = Path(root)
    t = _tables(data)
    best, transfer = data["training"]["fa_best"], data["transfer"]
    recommendation, roadmap = _recommendation(data)
    limitations = _limitations(data)
    commit = data["freeze"]["commit"]
    freeze_link = f"https://github.com/ReloadLightly/evopolis/commit/{commit}"
    effect = transfer["scores"]["BC1"]["surplus"]["E1"]
    level = transfer["scores"]["BC1"]["surplus"]["E2"]
    observed_prop = transfer["human"]["Proportional"]["surplus"]["mean"]
    boundaries = data["training"]["boundary_minima"]
    boundary_text = ", ".join(boundaries) if boundaries else "none"
    observed_fractions = {r["mechanism"]: r["observed_fraction"] for r in data["diagnostics"]["per_rule"] if r["family"] == "T03-GRU"}
    attenuation = {r["family"]: r for r in data["diagnostics"]["attenuation"]}
    attenuation_text = (f"The human validation Proportional−Equal choice-mean contrast is only {number(attenuation['T03-GRU']['observed_P_minus_E'], 4)}; "
                        + ", ".join(f"{label(family)} has index {number(attenuation[family]['attenuation_index'])}"
                                    for family in ("T03-GRU", best, "CL-" + best))
                        + ". Above-one ratios indicate amplification of this small validation contrast. The opened-test audit uses a different group subset and allocation-weighted fractions.")
    mc_text = "; ".join(label(f) + " " + number(transfer["scores"][f]["surplus"]["E1"]["prediction_mc_se"])
                        + "/" + number(transfer["scores"][f]["surplus"]["E2"]["prediction_mc_se"])
                        for f in transfer["primary_simulators"])
    d1_survival = []
    for family in (best, "CL-" + best):
        candidate = next(row for row in data["decision"]["candidates"] if row["family"] == family)
        comparison = candidate["comparisons"]["survival"]
        own, incumbent = comparison["candidate_ie"], comparison["bc1_ie"]
        d1_survival.append(f"{label(family)} {number(own['estimate'])} {bounds(own['ci95'])}, "
                           f"matched BC1 {number(incumbent['estimate'])} {bounds(incumbent['ci95'])}")
    d1_survival_text = "; ".join(d1_survival)
    audit_text = _audit_paragraph(data)
    shift = transfer["cohort_shift"]
    constraint = next(r for r in data["decision"]["human_population_constraints"] if r["population"] == "validation")
    validation_survival = constraint["metrics"]["survival"]["human_means"]
    if (constraint["strict_order_condition_feasible"] or validation_survival["Mixed"] != validation_survival["Proportional"]):
        raise RuntimeError("Observed D1 constraint differs from the audited design; manuscript wording must be reviewed")
    tie_text = (f"Validation survival ties Mixed and Proportional at {number(validation_survival['Mixed'])}; validation surplus orders Equal < Proportional < Mixed. "
                "Consequently all CL candidates are mechanically ineligible under the strict-order criterion, regardless of their IE. This is a constraint of the declared decision rule, not evidence of inaccurate calibrated behavior.")
    conclusion = "; ".join(label(f) + ": E1 " + transfer["D2"][f]["E1"] + ", E2 " + transfer["D2"][f]["E2"]
                           for f in (best, "CL-" + best))
    report = f"""# Institutional validity: do learned human models predict which allocation rule works?

## 1. Question and tipping point

> {QUESTION}

Four residents receive allocations from a 200-unit common pool, return integer amounts, and retain the rest. Returns grow by 1.4, capped at 200. With full allocation and below the cap, the next pool is 1.4Σc. Nonshrinkage requires the allocation-weighted fraction Σc/Σe ≥ 1/1.4 = 5/7; this is not an instantaneous death threshold or the arithmetic mean of individual return fractions. Small choice-distribution errors near this feedback threshold can produce large collective errors while changing likelihood little.

## 2. Simulators

Frozen references comprise Task 03 constant, linear, feedforward and GRU models; continued Task 04 feedforward and GRU models; and Task 05 P0/P1/H0/H1 controls. FA-GRU, FA-P0 and FA-H0 add the public offer-versus-previous-contribution slope and its undefined indicator. CL variants tilt legal PMFs by exp((τ0+τ1s)c/e). All learned lines use seeds 17/29/43; H families retain one resident effect throughout each game. We distinguish fitted parameters, deterministic history updates and posterior inference; no online parameter learning or evolutionary search occurs.

Recorded BC1 is the incumbent. The paper identifies its fitting evidence as “Based on Train Set 1” (537 games). BC1 predates Interpolating; BC2's 990-game collection covers “all aforementioned data”, described as “a total of 990 games”, so BC2 is descriptive. These are recorded upstream outcomes, not recreated networks. [Methods]({PAPER})

Each learned checkpoint forecasts 512 games under Equal, Mixed, Proportional and Interpolating, with no human prefix: 1,536 games per family/rule. Means use all 40 rounds, with zero padding after exact-zero termination; human residual floors remain recorded. Surplus, survival (pool40>1), Gini, pool20 and depletion time are group outcomes. Distributional fidelity uses energy distance on (surplus/10,pool20/200,pool40/200), excluding within-sample diagonals.

## 3. Experiment 1 benchmark

The table includes all 40 human groups per rule, including already opened test groups. Separate 32-group training+validation comparisons, survival, inequality, pool20 and energy distances are in the [benchmark artifact](../results/task06/benchmark.json).

{t['benchmark']}

![Experiment 1 institution benchmark.](assets/task06-institution-benchmark.png)

## 4. Attenuation and sensitivity

Diagnostics average nonforced choices within groups, then groups within rule and fitted seeds. Observed validation return fractions under Equal/Mixed/Proportional are {'/'.join(number(observed_fractions[r]) for r in RULES)}. Attenuation divides the predicted Proportional−Equal fraction difference by the observed difference.

{audit_text}

{t['attenuation']}

{attenuation_text}

![Validation attenuation.](assets/task06-attenuation.png)

The fixed sensitivity sweep applies τ∈{{−0.3,0,0.3,0.6,0.9}} with 256 games per checkpoint/rule/point. This slice reports Mixed at an additional τ=0.6; CL already includes its selected tilt. The figures show validation NLL cost against collective outcomes and mark the resource-weighted 5/7 reference separately from group-balanced fractions.

{t['sensitivity']}

![Tipping-point sensitivity.](assets/task06-tipping-sensitivity.png)

## 5. FA and CL results

All nine fresh FA fits complete 480 epochs on the existing 96/32 train/validation split. Earliest validation minima select checkpoints; **FA-best is {best}**. Budget-boundary minima: {boundary_text}. Boundary selection does not establish convergence. Persistent fits pass the declared 41-versus-81-node checks. Feature clipping affects {data['feature_audit']['clipped_rounds']}/{data['feature_audit']['rounds']} development rounds: {', '.join(f"{rule} {data['feature_audit']['by_rule'][rule]['clipped']}/{data['feature_audit']['by_rule'][rule]['rounds']}" for rule in (*RULES, 'M1'))}.

{t['fits']}

Calibration searches the complete declared coarse grid and 5×5 refinement per seed, using only training-group outcome distributions and common random numbers. It targets human fidelity, not higher surplus. Validation cost is calibrated minus base NLL over all 32 groups, including recorded M1 offers; seed-level tilts and costs remain in the artifacts.

{t['calibration']}

## 6. D1: development headroom

**D1: {transfer['D1']}.** IE averages absolute surplus errors equally over the three rules. Uncalibrated candidates use 32 development groups/rule; CL uses eight validation groups/rule. BC1 is rescored on exactly the same groups. Qualification requires matching both strict human surplus and survival orderings and IE≤BC1. Intervals resample complete human groups, holding forecasts fixed.

{t['D1']}

Survival-IE analogues with 95% group-bootstrap intervals are {d1_survival_text}. All candidates' survival errors and ordering checks are in the [D1 artifact](../results/task06/headroom_decision.json).

{tie_text}

## 7. Experiment 2 transfer and D2

The [public forecast freeze]({freeze_link}) preceded any Experiment 2 behavioral parsing. The cohort has 40 Proportional, 40 Interpolating and 40 recorded M1 groups. Observed surplus is {number(observed_prop)} under Proportional and {number(level['observed'])} under Interpolating: I−P={number(effect['observed'])}, human-group 95% interval {bounds(effect['observed_ci95'])}.

E1 is absolute error in I−P; E2 is Interpolating level error. The primary simulators are BC1, {best} and {label('CL-'+best)}. Task 03 GRU and constant are named secondary references. Paired differences subtract BC1's absolute error on the identical 2,000 bootstrap resamples, stratified by rule. Negative intervals excluding zero indicate better prediction; positive intervals excluding zero indicate worse prediction.

{t['transfer']}

**D2:** {conclusion}. Monte Carlo SEs, reported separately as effect/Interpolating-level SE, are {mc_text}. These use within-checkpoint variation with fixed seed weights.

![Experiment 2 effects, levels and paired comparisons.](assets/task06-transfer-effects.png)

Secondary survival contrasts, Proportional level error, energy and pool20 errors, and per-seed results remain in the [transfer artifact](../results/task06/transfer.json).

All 57 learned checkpoints/variants receive teacher-forced evaluation on all 120 groups; M1 uses recorded current offers only. NLL averages nonforced choices within groups, then groups and fitted seeds equally; the all-120 column averages three equally sized rule strata. The table gives primary learned lines and named references; full per-seed NLL and outcome results are machine-readable.

{t['nll']}

Descriptive Exp2−Exp1 surplus shifts are {number(shift['Proportional']['Exp2_minus_Exp1']['surplus'])} for Proportional and {number(shift['M1']['Exp2_minus_Exp1']['surplus'])} for M1. These comparisons do not reuse people as paired observations.

## 8. Limitations

{limitations}

## 9. Recommendation

{recommendation}

## 10. Paper readiness

The defensible claim concerns measured institutional forecast validity and the gap between individual likelihood and collective fidelity. Baselines include recorded BC1, an input-blind constant, neural and conditional controls, and the simple FA/CL ingredients. A paper needs broader independent human evidence and instruction-matched replication; executable upstream models would improve comparability. An LLM-agent baseline is future work. This study does not establish motives, first discovery of the local-versus-collective discrepancy, or evolutionary improvement.

The [source audit](../results/task06/institutional_source_audit.json), [frozen manifest](../results/task06/manifest.json), [transfer results](../results/task06/transfer.json) and [phase resources](../results/task06/runtime.json) carry provenance. Earlier reports cover [behavioral fitting](behavioral-agents.md), [collective forecasts](collective-forecast-fidelity.md) and [conditional responses](conditional-responses.md).
"""
    existing = (root / "README.md").read_text()
    cover = re.search(r'<p align="center">\s*<img src="docs/assets/evopolis-cover-ffv\.png".*?</p>', existing, re.S)
    if cover is None or not (root / "docs/assets/ffv-community-viewer.png").is_file():
        raise RuntimeError("Report refused; existing cover or preserved viewer screenshot is missing")
    readme = f"""{cover.group(0)}

# EvoPolis: can learned human models predict which institution works?

## Abstract

We test whether behavioral models fitted to human common-pool decisions predict collective outcomes and differences between allocation rules, including an unseen rule. We compare recorded DeepMind BC1 simulations with neural and conditional models, a public institutional-response feature, and calibration to human outcome distributions. Nine new fits and 19 forecast families support a publicly frozen Experiment 2 transfer evaluation. D1 reports **{transfer['D1']}**. {conclusion}. The research plots report measured results. **No evolutionary search has run**; Experiment 3 remains closed.

## Introduction

> {QUESTION}

Institutions alter incentives and the trajectories of shared resources. A useful learned social world model must predict these changes, not merely assign reasonable probabilities to individual decisions. EvoPolis begins with the published four-person common-pool experiment of Koster, Pîslar and colleagues. Reproducing recorded outcomes, fitting new behavioral models and extending their evaluation are separate activities: these are new EvoPolis fits, while BC1/BC2 numbers are recorded upstream simulations.

Each round an allocation rule distributes resources, residents return contributions and keep the remainder. The scientific target is agreement with human behavior under each rule and with human differences between rules. Prosperous simulated inhabitants alone do not validate a model.

## Data and methods

The official release supplies human group trajectories and recorded behavioral-clone games. Experiment 1 provides 160 groups across Equal, Mixed, Proportional and M1, split into 96 training, 32 validation and 32 previously opened diagnostic test groups. Parameters and checkpoints use training/validation evidence only. Experiment 2 supplies 120 new groups across Proportional, Interpolating and M1. Forecasts, checkpoints, calibrated tilts, seed tables and analysis hashes were [committed and pushed]({freeze_link}) before opening its behavioral rows.

The published environment starts with 200 units, has four residents, and runs for 40 rounds. Allocations may be fractional; new models choose legal integers. With allocation e and contributions c, the next pool is min(200,R−Σe+1.4Σc). Under full allocation below capacity, nonshrinkage requires the allocation-weighted return fraction Σc/Σe≥5/7. This feedback threshold is not instantaneous depletion, and it differs from an unweighted mean of individual return fractions.

References include Task 03 constant/linear/feedforward/GRU, continued Task 04 feedforward/GRU, and Task 05 P0/P1/H0/H1. Three fresh FA families add a public slope relating current offer shares to previous contribution shares, with an undefined indicator. FA-best minimizes mean validation NLL across three starts. CL calibrates six lines with a legal exponential tilt, matching training-group outcome distributions instead of targeting higher surplus. Persistent resident effects are drawn once per game. History and posterior updates occur under frozen parameters; they are distinct from parameter training.

Every learned checkpoint simulates 512 games per rule, pooling 1,536 games over seeds 17/29/43. Recorded BC1 supplies 512 games per rule. Outcomes include 40-round surplus, Gini across four player means, survival (pool40>1), pool20 and depletion time. Energy U statistics compare (surplus/10,pool20/200,pool40/200). Intervals resample human groups 2,000 times; Monte Carlo uncertainty is reported separately. M1/M2 have no executable policy and support only recorded outcomes or teacher-forced offers. [Full protocol](docs/protocol.md) and [data inventory](docs/data-inventory.md).

## Results

### Experiment 1 and the incumbent benchmark

{t['benchmark']}

Human rows include all 40 groups per rule, including opened test groups. The [full benchmark](results/task06/benchmark.json) also evaluates training+validation groups alone and records survival, inequality, energy distances and institutional ordering.

![Experiment 1 institution benchmark.](docs/assets/task06-institution-benchmark.png)

### Tipping-point diagnosis and simple ingredients

The human validation Proportional−Equal choice-mean contrast is only {number(attenuation['T03-GRU']['observed_P_minus_E'], 4)}; {best}'s index is {number(attenuation[best]['attenuation_index'])}. Ratios above one indicate amplification of this small contrast. The sensitivity curves compare changes in validation likelihood with changes in simulated survival and surplus; the fraction panels mark 5/7 with its resource-weighting caveat.

![Validation response attenuation.](docs/assets/task06-attenuation.png)

{t['sensitivity']}

![Measured tipping-point sensitivity.](docs/assets/task06-tipping-sensitivity.png)

All nine FA fits complete 480 epochs. **FA-best is {best}**; mean validation NLLs are {', '.join(f'{family} {number(value, 4)}' for family, value in data['training']['mean_validation_nll'].items())}. Budget-boundary minima: {boundary_text}. The selected {label('CL-'+best)} validation-NLL cost is {number(next(r['cost'] for r in data['nll_cost']['per_family_all_rules'] if r['family'] == best), 4)} nats. All selected tilts and numerical checks are reported in the [study](docs/institutional-validity.md).

**D1: {transfer['D1']}.** The decision compares surplus IE against BC1 on identical human groups and requires both strict human surplus and survival ordering. {tie_text}

### Frozen Experiment 2 transfer

Human surplus is {number(observed_prop)} under Proportional and {number(level['observed'])} under Interpolating. Their difference is {number(effect['observed'])}, with 95% human-group interval {bounds(effect['observed_ci95'])}. E1 tests this within-cohort difference; E2 tests the Interpolating level.

{t['transfer']}

The first three rows are primary; the last two are named secondary references. Paired intervals compare absolute errors with BC1 using the same human resamples. **D2:** {conclusion}. Separate effect/level Monte Carlo SEs are {mc_text}.

![Experiment 2 observed and predicted effects with intervals.](docs/assets/task06-transfer-effects.png)

All 57 learned checkpoints/variants were also scored on all 120 groups, with M1 likelihood separate. NLL averages nonforced choices within groups, then groups and fitted seeds equally. Survival contrasts, Proportional level errors, pool20 errors, energy distances, per-seed results and descriptive cohort shifts are in the [transfer artifacts](results/task06/transfer.json) and [full report](docs/institutional-validity.md).

### Roadmap and observatory

{roadmap}

![Existing EvoPolis viewer showing a recorded four-resident community.](docs/assets/ffv-community-viewer.png)

The original 16-bit observatory displays recorded and learned trajectories. The cover is concept art; this image is an existing application screenshot. Task 06 changes no interface. Earlier evidence is documented in [behavioral agents](docs/behavioral-agents.md), [collective forecast fidelity](docs/collective-forecast-fidelity.md) and [conditional responses](docs/conditional-responses.md).

## Limitations

{limitations}

## References

Koster, R., Pîslar, M., et al. (2025). [Deep reinforcement learning can promote sustainable human behaviour in a common-pool resource problem]({PAPER}). Nature Communications 16, 2824. BC1 used Train Set 1; BC2 includes earlier experimental evidence and is descriptive here.

[Official DeepMind release, pinned revision 4f1a99a](https://github.com/google-deepmind/sustainable_behavior/tree/4f1a99a9d150f9fa6bad1a0f70f11c0673d46763). Adapted source retains attribution; see [source notes](docs/SOURCES.md), the [Task 06 specification](docs/tasks/06-institutional-validity.md) and [preregistered manifest](results/task06/manifest.json).
"""
    if _words(report) > 2500:
        raise RuntimeError(f"Report exceeds 2,500 words: {_words(report)}")
    if _prose_words(readme) > 1800:
        raise RuntimeError(f"README prose exceeds 1,800 words: {_prose_words(readme)}")
    outputs = {"docs/institutional-validity.md": report, "README.md": readme,
               "docs/github-description.txt": DESCRIPTION + "\n"}
    if transfer["D1"] == "headroom":
        outputs["docs/tasks/07-evolved-behavioral-programs.md"] = _task07(data)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate and render in memory without changing requested documents")
    args = parser.parse_args()
    data = evidence()
    outputs = render(data)
    if not args.check:
        for name, text in outputs.items():
            path = ROOT / name
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".partial")
            temporary.write_text(text)
            temporary.replace(path)
    print(json.dumps({"written": not args.check, "outputs": list(outputs),
                      "report_words": _words(outputs["docs/institutional-validity.md"]),
                      "readme_prose_words": _prose_words(outputs["README.md"]),
                      "D1": data["transfer"]["D1"], "D2": data["transfer"]["D2"]}, indent=2))


if __name__ == "__main__":
    main()
