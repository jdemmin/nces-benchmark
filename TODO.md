# Naming

- GLOSSARY.md §9 pins complexity to ``DL-expression length`` and forbids ``difficulty``. Since its now multi-dimensional, that entry needs rewriting. Probably complexity becomes the structured object, dl_length the old scalar.
- ``Hardness`` or ``separability`` are defensible; the glossary should pick one and enforce it.

# Cost and independence

- The extensional fields need reasoner calls over target concepts. Already computed target extensions during evaluation, so cache and reuse them rather than computing twice. More importantly: best_atomic_f1 must be computed from the knowledge base alone, never from anything embedding-derived, or it contaminates the independent variable.
- _concept_complexity currently counts ``.`` as an atom-excluded token but ≤/≥ as constructors, while ALCHIQ(D) cardinality restrictions render with numerals that will be counted as atoms. The existing complexity values are already slightly wrong under rho: ALCHIQD. Not necessary problematic yet since ALC only.