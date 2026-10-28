# Overview: what SAI is

## The dual definition

**SAI (object).** The Science of AI is the body of knowledge about machine-learning systems whose every claim carries a machine-checkable applicability boundary, and that is organized into three epistemic levels — knowledge, law, science — defined in [ladder.md](ladder.md).

**SAI (system).** A Science-of-AI system is an agent that produces and maintains such a body of knowledge: it answers questions from its knowledge base, converts failures into claims, verifies claims on fresh experiments, consolidates the verified into laws and models, and chooses what to run next.

Same word, two senses, one road: the object defines the target format, the system is the machine that manufactures it. *The ladder assigns the grade; the loop earns it.*

## Why a new noun

"AI for science" names a customer (science) and a supplier (AI). The needed term names an **object**: the science whose subject matter is AI itself. Physics studies matter; SAI studies learning machines — an instrumentally exhaustible subject (every hypothesis can in principle be settled by running an experiment) that today lacks the institution to do so systematically.

This follows a 2025–2026 convergence that named the same object from different angles: *The Science of AI* lecture notes (Bahri), *On the Science of 'AI'* (Kambhampati), *AI as Normal Science* (Narayanan–Kapoor), *The Science of Artificial Intelligence* (DeDeo), *The Science of LLMs* (Wang), *On the Science of Alien Intelligence* (Tang). The taxonomy differs from all of them in one move: its levels come with **admission gates that a machine can check**, so classification is a procedure, not a taste.

## What counts as the object of study

The subject matter is *the training and use of learning machines*: architectures, optimizers, data, objectives, budgets, and the behaviors that emerge. The distinguishing property: hypotheses here are unusually cheap to *state* and unusually expensive to *settle* — settling one costs a real training run. That expensive-oracle property shapes the whole design of the loop (see [loop.md](loop.md)).

## Relation to neighboring terms

| Term | What it covers | Relation to SAI |
|---|---|---|
| Empirical ML | methods, recipes, benchmarks | the field's volume; mostly L1 raw material |
| Learning theory | NTK, statistical mechanics, RL theory | L3 instruments, often under assumptions too strong to deploy |
| Scaling-law research | fitted quantitative relations | the clearest L2 exemplar family |
| AI-scientist systems | agents that run research loops | producer axis; most are L1-producing instruments today |
| Benchmark science | measuring task performance | reads L1 only — never knowledge or ascent (see [milestones.md](milestones.md)) |

The taxonomy does not demote neighbors; it grades their **output** and asks what gates the output has passed.
