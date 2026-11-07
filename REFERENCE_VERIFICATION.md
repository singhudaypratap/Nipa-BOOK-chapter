# Reference verification notes

The 25-entry reference list supplied with the instructions was **not used verbatim**.
Crossref and doi.org were unreachable from the build environment (proxy policy), so each DOI
could not be resolved automatically. Web searches for several supplied entries found no matching record:

- `10.1109/JIOT.2022.3204112` (Selvaraj, Singh, Kumar)
- `10.1016/j.envsoft.2022.105610` (Singh & Siddiqui)
- `10.1016/j.resconrec.2021.106041` (Boning et al.)
- `10.1038/s41558-023-01890-w` (Zhang & Sun)
- `10.1109/TII.2022.3189901` (Schlosser et al.), `10.1016/j.cma.2022.115790`, `10.1016/j.enbuild.2022.112640`,
  `10.1029/2022WR033100`, `10.1029/2021GL097100` and `10.1093/acrefore/9780190228620.013.821`
  (the last is attached to a book title it does not belong to) are likewise unconfirmed.

These were replaced with well-known, widely cited works whose DOIs are standard (e.g. ERA5, PINNs,
GraphCast, SHAP, FourCastNet, CMIP6 overview). Kept from the supplied list: Grieves & Vickers, Raissi et al.,
Kashinath et al., Pathak et al., Lam et al., Hersbach et al., Rasp et al., Lundberg & Lee, Evensen (book),
Eyring et al., Shi et al., Dosovitskiy et al., Gal & Ghahramani.

**Before submission, resolve every DOI at https://doi.org/ and confirm title/authors/pages.**
In particular, check the author list and page range of each entry against the publisher record.

Build: `python3 tools/build_chapter.py <ChapterDir>` (numbers citations in order of appearance and fails on
unknown or uncited keys). Figures: `python3 <ChapterDir>/figures_code/make_figures.py`.
Word counts include table text; replace the ORCID placeholders before submission.
