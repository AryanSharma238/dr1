# Merged-twin cosmic string finder (Euclid Q1 / DR1)

Core method
- csrc/hough.c      : compiled blind line scanner (weighted perpendicular-elongation statistic over all angles/positions). Build: gcc -O3 -fPIC -shared -o csrc/libhough.so csrc/hough.c -lm
- blind.py          : Python wrapper + catalogue prep (shape -> chi components, local-mean shape subtraction in 600" cells)
- weights.py        : optimal per-galaxy weights (expected response / shape-noise variance)
- imgmodel.py       : image-level string lensing of a galaxy (partial band duplication, PSF, adaptive moments)
- isomodel.py       : same, with SourceExtractor-like isophotal moments and noise
- field.py          : Q1 field loader + gnomonic projection

Searches & tests
- fullscan.py       : full blind scan of a Q1 field (data or shape-scrambled control)
- blindinj.py, merged3.py, pipeinj.py : injection/recovery tests
- vet.py            : kill tests on candidate lines (bright/faint split, half split, offset profile, brightness)
- extend.py         : trace a candidate great-circle line across the survey and re-score it
- pix/realval2.py   : validation on real Euclid VIS cutouts (response factor ~0.6)

DR1 pipeline
- tilecenters.py, dr1pipe.py, dr1run.py : streaming patch-by-patch DR1 scan (ESA archive, dr1.mer_catalogue_wide_survey)
- ka_restart.sh, ka_vet.py : keep-alive restart + automatic extension of patch peaks

Data tables
- completeness.npy  : measured catalogue close-pair completeness vs separation
- resp_table.npy    : response vs intrinsic galaxy size
