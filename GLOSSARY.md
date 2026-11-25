# Biomedical Target Dossier Synthesis

Automated retrieval of human gene biology, open-access literature, and structured experimental protocols to produce evidence-grounded biomedical research dossiers.

## Language

### Research Entities

**Target**:
The biological entity or topic under investigation—typically a human gene symbol (e.g., `VEGFA`, `TGFB1`) optionally narrowed by a biomaterial or experimental context.
_Avoid_: Query, subject, keyword, entity

**Gene Profile**:
Canonical human gene metadata fetched from NCBI Gene for a Target, comprising official symbol, Entrez Gene ID, description, aliases, chromosomal map location, and curated summary.
_Avoid_: Gene info, Entrez dump, annotation record

**Mined Paper**:
A literature record returned from Semantic Scholar (preferring open-access works) containing bibliographic metadata, abstract, identifiers (DOI/PMID), citation count, and its optional Extracted Methodology.
_Avoid_: Article, hit, publication, citation

**Extracted Methodology**:
Structured experimental parameters parsed from a Mined Paper's abstract across eight fixed categories: cell lines, biomaterials, centrifuge speeds, curing times, concentrations, temperatures, outcomes, and other parameters.
_Avoid_: Protocol dump, metadata, parameters

### Synthesis & Conversation

**Executive Dossier**:
The structured six-section Markdown synthesis compiled for a Target from its Gene Profile and Mined Papers: Executive Summary, Gene Overview, Recent Research Findings, Extracted Methodology Comparison, Key Contradictions & Research Gaps, and References.
_Avoid_: Report, summary, briefing, write-up

**Protocol Conflict**:
An explicit disagreement across Mined Papers in experimental conditions, biomaterial formulations, concentrations, or reported outcomes for the same Target.
_Avoid_: Bug, mismatch, inconsistency

**Research Gap**:
An experimental parameter, biological mechanism, or evidence category absent from the retrieved Gene Profile and Mined Papers, surfaced explicitly rather than inferred.
_Avoid_: Missing field, null value, limitation

**Research Session**:
The accumulated conversational context for an interactive investigation—active Target, Gene Profile, Mined Papers, latest Executive Dossier, and Tool Trace.
_Avoid_: Thread, chat memory, workspace

**Tool Trace**:
The ordered record of external database lookups and synthesis invocations executed during a conversational turn, including inputs, outcomes, and surfaced errors.
_Avoid_: Activity log, debug trail, stack trace
