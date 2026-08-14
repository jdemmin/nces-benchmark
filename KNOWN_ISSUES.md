# Known Issues

## ontolearn learning_problem_generator.get_examples() is broken

`ontolearn==0.10.0` declares `owlapy==1.6.3` as its exact required dependency,
so this is not a project-side version-pinning issue. It is an upstream defect:
`LearningProblemGenerator.get_examples()` reads `example_node.concept.instances`,
but `.instances` is only ever set on the `RL_State` wrapper itself
(`next_rl_state.instances = set(self.kb.individuals(next_rl_state.concept))`
in `apply_rho_on_rl_state`), never on `.concept` (a plain `OWLClassExpression`).
This raises `AttributeError: 'OWLClass' object has no attribute 'instances'`.
