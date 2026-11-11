#!/bin/bash
cd /home/claude/tf-exp
for S in testA-sherlock testB-studio-docs; do
  [ -f res/$S/m1-g32.npz ] || python3 run_eval.py $S 64 800 res/$S q8_0 q4_k_m random-row.tforge m1-g32.tforge >> log-$S-1.txt 2>&1
  [ -f res/$S/m4-row-heal-0.00001.npz ] || python3 run_eval.py $S 64 800 res/$S m2s-g32.tforge m3-row.tforge m4-row.tforge m4-row-heal-0.00001.tforge >> log-$S-2.txt 2>&1
done
echo ALLDONE >> log-tests-done.txt
