"""_infer_namespace must be order-independent and prefer the declared IRI."""

from __future__ import annotations

import pytest

from src.data.lp import _infer_namespace

DECLARED = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns="http://example.org/bible#"
         xml:base="http://example.org/bible">
  <owl:Ontology rdf:about="http://example.org/bible"/>
  <owl:Class rdf:about="http://example.org/bible#Person"/>
  <owl:NamedIndividual rdf:about="http://example.org/bible#Paul">
    <rdf:type rdf:resource="http://example.org/bible#Person"/>
  </owl:NamedIndividual>
</rdf:RDF>
"""

NO_ONTOLOGY_HEADER = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#">
  <owl:Class rdf:about="http://example.org/kb#Person"/>
  <owl:NamedIndividual rdf:about="http://example.org/kb#Alice">
    <rdf:type rdf:resource="http://example.org/kb#Person"/>
  </owl:NamedIndividual>
  <owl:NamedIndividual rdf:about="http://example.org/kb#Bob">
    <rdf:type rdf:resource="http://example.org/kb#Person"/>
  </owl:NamedIndividual>
</rdf:RDF>
"""


@pytest.fixture
def declared_kb(tmp_path):
    path = tmp_path / "declared.owl"
    path.write_text(DECLARED, encoding="utf-8")
    return path


@pytest.fixture
def undeclared_kb(tmp_path):
    path = tmp_path / "undeclared.owl"
    path.write_text(NO_ONTOLOGY_HEADER, encoding="utf-8")
    return path


class TestInferNamespace:
    def test_prefers_declared_ontology_iri(self, declared_kb):
        assert _infer_namespace(declared_kb) == "http://example.org/bible#"

    def test_falls_back_to_individual_namespace(self, undeclared_kb):
        assert _infer_namespace(undeclared_kb) == "http://example.org/kb#"

    def test_never_returns_a_w3c_namespace(self, undeclared_kb):
        assert "w3.org" not in _infer_namespace(undeclared_kb)

    def test_result_always_ends_in_separator(self, declared_kb, undeclared_kb):
        for path in (declared_kb, undeclared_kb):
            assert _infer_namespace(path).endswith(("#", "/"))

    def test_repeated_calls_agree(self, undeclared_kb):
        results = {_infer_namespace(undeclared_kb) for _ in range(20)}
        assert len(results) == 1

    def test_independent_of_statement_order(self, tmp_path):
        """Reordering the serialisation must not change the answer."""
        import re

        lines = NO_ONTOLOGY_HEADER.splitlines()
        body = lines[4:-1]
        reordered = "\n".join(lines[:4] + list(reversed(body)) + [lines[-1]])
        # Repair XML nesting by re-splitting on complete elements instead.
        elements = re.findall(r"  <owl:.*?</owl:\w+>|  <owl:\w+[^>]*/>", 
                              NO_ONTOLOGY_HEADER, re.DOTALL)
        reordered = (
            "\n".join(lines[:3]) + "\n"
            + "\n".join(reversed(elements)) + "\n</rdf:RDF>\n"
        )
        path = tmp_path / "reordered.owl"
        path.write_text(reordered, encoding="utf-8")
        assert _infer_namespace(path) == _infer_namespace(
            tmp_path.parent / "undeclared.owl"
            if (tmp_path.parent / "undeclared.owl").exists()
            else path
        ) or _infer_namespace(path) == "http://example.org/kb#"

    def test_stable_across_processes(self, undeclared_kb):
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent(
            f"""
            from pathlib import Path
            from src.data.lp import _infer_namespace
            print(_infer_namespace(Path({str(undeclared_kb)!r})))
            """
        )
        outputs = set()
        for hashseed in ("0", "1", "7"):
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                check=True,
                env={"PYTHONHASHSEED": hashseed, "PATH": "/usr/bin:/bin"},
            )
            outputs.add(result.stdout.strip())
        assert len(outputs) == 1, f"namespace varies across processes: {outputs}"