@echo off
cd /d d:\tung\avora-matching\AVORA-MATCHING
echo LAUNCHED %DATE% %TIME% > _run_marker.txt
streamlit run dashboard.py > _streamlit2.log 2>&1
echo EXITED errorlevel=%ERRORLEVEL% >> _run_marker.txt
