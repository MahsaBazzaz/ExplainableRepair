# Guiding Level Repair with Explainable Playability Classifiers
Procedurally generated levels created by machine learning models can be unplayable without any further editing. There have been various approaches to automatically repairing generated levels. However, such constraint-based repairs become slow, especially as levels grow larger. This paper proposes utilizing explainability methods on playability classifiers to identify the level regions that contribute to unplayability, and use this information to provide weights to constraint-based level repairer.  Our results, across three games, show that when using model explainability we can repair procedurally generated levels faster.

# System Description
![](./doc/system_overview.png)
The main idea is that an explainable playability classifier will provide relative attributions for how much each tile contributes to the classification, and these attributions will be converted into penalty weights to guide the repair, such that tiles with **more** attribution to the classification have a **lower** weight and are thus less penalized for changing during the repair.  Ideally, this weighting information can be incorporated by the repairer to guide it to focus on locations that are more helpful to change, and thus more efficiently repair the level.


