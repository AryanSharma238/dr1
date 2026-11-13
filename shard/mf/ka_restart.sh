cd /home/claude/strings
if [ $(ps -o args -C python3 | grep -c "dr1run.py") -lt 2 ]; then
  pkill -f dr1run.py; pkill -f dr1pipe.py; sleep 1
  for d in dr1/P*; do [ -f $d/summary.json ] || rm -rf $d; done
  export OMP_NUM_THREADS=1
  nohup python3 dr1run.py dr1.mer_catalogue_wide_survey 1000000 2 0 >> dr1run_a.log 2>&1 &
  nohup python3 dr1run.py dr1.mer_catalogue_wide_survey 1000000 2 1 >> dr1run_b.log 2>&1 &
  echo restarted
fi
echo "done patches: $(ls dr1/P*/summary.json 2>/dev/null | wc -l)/270"
