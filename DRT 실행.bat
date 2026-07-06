@echo off
chcp 65001 >nul
title DRT 분석 프로그램
cd /d "%~dp0"

echo ============================================
echo   DRT 분석 프로그램 (Distribution of RT)
echo ============================================
echo.

python --version >nul 2>&1
if errorlevel 1 goto NOPY

python -c "import streamlit, plotly, numpy, scipy, pandas" >nul 2>&1
if errorlevel 1 goto INSTALL
goto RUN

:INSTALL
echo [설치] 필요한 패키지를 설치합니다. 잠시만 기다리세요...
python -m pip install --quiet streamlit plotly numpy scipy pandas
if errorlevel 1 goto PIPFAIL
goto RUN

:RUN
echo 브라우저에서 앱이 열립니다. 종료하려면 이 창에서 Ctrl+C 를 누르세요.
echo.
python -m streamlit run app.py
goto END

:NOPY
echo [오류] Python 을 찾을 수 없습니다. Python 3 을 설치하고 PATH 에 추가하세요.
echo        https://www.python.org/downloads/
goto END

:PIPFAIL
echo [오류] 패키지 설치에 실패했습니다.
goto END

:END
echo.
pause
