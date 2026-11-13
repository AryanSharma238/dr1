# DR1 cosmic-string scan on GitHub Actions (no terminal needed)

WHAT IT DOES: 20 cloud machines x 4 workers each scan the 270 DR1 patches with your merged-twin pipeline (fast scanner, angle step 0.30),
then a final job extends every line above S=6.8 (up to 5 per patch). Expected ~1.5-2 h wall time; UNTESTED on GitHub, so do the test run first.

## 1. Create the repo (public)
1. github.com -> click "+" (top right) -> "New repository".
2. Name: dr1-string-scan.  Choose PUBLIC (free Actions minutes + 4-core machines).  Leave README/.gitignore/license OFF.  Click "Create repository".

## 2. Upload the files (web page, no git)
1. On the empty repo page click "uploading an existing file" (or Add file -> Upload files).
2. Unzip dr1_shard_kit.zip on your computer. Open the unzipped "shard" folder. Select EVERYTHING inside it
   (shard.sh, dr1pipe.py, hough_fast.c, ka_vet_all.py, tilecenters_dr1.csv, README.md, WORKFLOW_PASTE_ME.yml and the folder "mf") and drag it into the browser box.
   Drag the CONTENTS of shard, not the shard folder itself, so shard.sh ends up at the top level of the repo.
3. Wait until all files show, scroll down, click "Commit changes".

## 3. Create the workflow file by hand (the hidden .github folder does not upload by drag-and-drop)
1. Add file -> Create new file.
2. In the name box type exactly:  .github/workflows/dr1-scan.yml   (typing each "/" makes a folder).
3. Open WORKFLOW_PASTE_ME.yml from the kit in any text editor, select all, copy, paste into the big box.
4. Click "Commit changes" -> "Commit changes".

## 4. (Optional) skip patches you already finished
Without this the first ~16 patches get redone (about 6% extra work, harmless). Only use it if tilecenters_dr1.csv in the repo is the SAME file your
earlier run used (patch numbers P0000.. depend on it). Add a text file done_patches.txt (Add file -> Create new file), one tag per line: P0000, P0003 ...

## 5. Test run (one machine, one patch per worker)
1. Click the "Actions" tab. If a green button asks to enable workflows, click it.
2. Left side: "dr1-string-scan" -> "Run workflow" (right) -> set only_shard = 0, max_patches = 1 -> green "Run workflow".
3. Wait ~30-40 min. Click the run -> scroll to "Artifacts" -> download dr1-shard-0 -> open run_0.log .. run_3.log.
   GOOD: "DR1 tiles with centres: 8947", "patches: 270", "[P....] N tiles in patch", "downloaded ... galaxies", then a line with max_S.
   BAD: Python traceback, "tile ... failed", or empty downloads (see Troubleshooting).

## 6. Full run
Actions -> dr1-string-scan -> Run workflow -> leave both boxes as they are -> Run. 20 jobs start. Each stops itself at 330 min and still uploads results.

## 7. Results
After all jobs finish, Artifacts at the bottom of the run page:
- dr1-vet-results -> dr1_results.txt: one block per patch (tag, max_S, and for each strong line its RA/Dec/PA and the extension "total S").
  Your earlier validation: real strings extend to S ~9-12 over 6 deg. Anything else still needs your kill tests.
- dr1-shard-N: summary.json per patch (full line lists) and summary_scr*.json (shape-scrambled controls = noise level) and the run logs.
Artifacts are deleted after 3 days (set in the workflow) - download them first.

## Troubleshooting
- No "Run workflow" button: the workflow file is missing or misnamed. Redo step 3 (path must be exactly .github/workflows/dr1-scan.yml).
- Many "tile ... failed"/empty downloads: ESA throttling. Wait and rerun, or test with fewer machines.
- Machine hit 330 min: normal stop; finished patches are uploaded. Add their tags to done_patches.txt and run again (the workflow does not resume by itself).
- pip/gcc errors: copy the red error text from the job log and send it to me.

## Privacy
This repo is public and artifacts contain candidate line positions. Download the results, then delete the artifacts (or the repo / switch it to private in Settings).
