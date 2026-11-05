Context-Aware Multi-Agent Architecture Overview
To generate publication-ready book chapters that comply with the editor's strict word count (7,000–10,000 words), Turnitin plagiarism threshold (<10% overall, <2% single-source), zero AI plagiarism, verified IEEE references with valid DOIs, and high-resolution black-and-white PowerPoint-style figures (min 1500×1500 px, 12pt font), a 5-agent pipeline is required.
┌────────────────────────────────────────────────────────────────────────┐
│                        AGENT 1: ARCHITECT                              │
│   Allocates word count budget, structural breakdown, and metadata     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        AGENT 2: CITATION GROUNDING                     │
│   Retrieves & verifies 20+ real DOIs via Crossref/Semantic Scholar     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        AGENT 3: CORE TECHNICAL WRITER                  │
│   Writes 7,500–9,000 words of technical content, equations, models     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        AGENT 4: B&W VISUALIZATION ARCHITECT            │
│   Generates 1500+ px B&W PowerPoint figures (12pt font) & IEEE tables  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        AGENT 5: EDITORIAL COMPLIANCE                   │
│   Audits citations, cross-references, similarity, and formatting       │
└────────────────────────────────────────────────────────────────────────┘
