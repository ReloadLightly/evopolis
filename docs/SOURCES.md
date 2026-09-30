# Source and asset notes

## Initial empirical reference

Raphael Koster, Miruna Pîslar, Andrea Tacchetti, Jan Balaguer, Leqi Liu, Romuald Elie, Oliver P. Hauser, Karl Tuyls, Matt Botvinick, and Christopher Summerfield. 2025. *Deep reinforcement learning can promote sustainable human behaviour in a common-pool resource problem*. Nature Communications 16, 2824. https://doi.org/10.1038/s41467-025-58043-7

- Official repository: https://github.com/google-deepmind/sustainable_behavior
- Inspected revision: `4f1a99a9d150f9fa6bad1a0f70f11c0673d46763`.
- Analysis notebook: `notebooks/sustainable_behavior.ipynb`.
- Dataset endpoint referenced by that notebook: https://storage.googleapis.com/sustainable_behavior/sustainable_behavior.csv
- Release scope: data access and figure/statistical analysis code. The inspected tree does not contain a complete simulator, model-training pipeline, or pretrained weights.
- Upstream software: Apache License 2.0. Other upstream materials: Creative Commons Attribution 4.0, according to the upstream README. Preserve the applicable notices with any imported material.
- Initial release inspection found both human experiment trajectories and pre-generated behavioral-clone rollouts. The original initial behavioral-cloning training cohort is not identifiable among the released condition labels. Do not equate the paper's full participant count with the available training population.
- Task 01 will preserve a downloaded checksum and reproducible inventory of the schema, human/synthetic conditions, group identifiers, and exclusions before analysis. No empirical reproduction is claimed in this initial documentation commit.

## Broader references

- Percy Liang. *Simulation: The Next Frontier for AI*. https://www.simile.com/blog/simulation-next-frontier
- Joon Sung Park et al. 2023. *Generative Agents: Interactive Simulacra of Human Behavior*. https://arxiv.org/abs/2304.03442
- StanfordHCI/genagents: https://github.com/StanfordHCI/genagents . The public demographic agents are not the restricted interview-based population described by the associated research.
- Mesa: https://mesa.readthedocs.io/stable/

## Original repository graphics

`assets/evopolis-cover.png` is AI-generated conceptual artwork created with the built-in image-generation tool. The exact generation prompt is preserved in `assets/cover-prompt.txt`. It is not a screenshot of working software, a representation of actual participants, or an empirical result.

`assets/research-loop.svg` and `assets/commons-cycle.svg` are editable vector diagrams of the proposed research architecture and resource mechanism. Their arrows represent dependencies and resource flows, not measured effects.

Source links and upstream licenses do not imply affiliation with or endorsement by the referenced institutions.
