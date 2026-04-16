# JacobiForcing — Codebase Explanation

**TL;DR.** Jacobi Forcing trains an autoregressive LLM to behave as a *causal parallel decoder* by finetuning it on noisy future blocks drawn from Jacobi fixed-point trajectories. The repo ships the full pipeline: bucketing, trajectory generation, noise-window preparation, noise-conditioned training, and a lightweight inference engine with rejection recycling. This document walks the method conceptually and points to the code that implements each stage.

## Contents

1. [What Jacobi Forcing is](#1-what-jacobi-forcing-is)
2. [The five method stages](#2-the-five-method-stages)
3. [The model-code layer](#3-the-model-code-layer)
4. [Repo map](#4-repo-map)
5. [Quirks and gotchas for reproduction](#5-quirks-and-gotchas-for-reproduction)

## Scope and conventions

This document is **concept-first**: each method stage is explained before its code pointer. It is *not* a directory reference — see §4 for that. All code paths are relative to the upstream repo root `hao-ai-lab/JacobiForcing` at the commit recorded in `docs/training_plan.md` Stage 1. Paths beginning with `JacobiForcing/` refer to the **nested subdirectory** inside the repo (see §5).
