# Dataset Notes

## Source
- NASA93 software effort dataset (Zenodo: https://zenodo.org/record/268419)
- File: data/raw/nasa93.arff

## Summary (from my own inspection)
- 93 projects, 24 columns, no missing values
- Each row is one completed NASA software project
- Target: act_effort (actual effort, person-months)
- Size feature: equivphyskloc
- Effort is highly skewed (median 252, mean 624, max 8211)

## Limitations
- Small dataset (93 rows), so results will be noisy
- Old projects (1971-1987) from NASA only, may not represent modern projects
- Contains no risk label, so the risk target must be defined and documented later (Week 3)

## Assumptions
- Effort multiplier ratings (l, n, h ...) will be converted to ordered numbers
- ID columns (recordnumber, projectname) will be dropped before training