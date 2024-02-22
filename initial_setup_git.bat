@echo off

REM Get git usernames and repository
echo Please Enter GitHub username :
set /p username=

echo Please enter GitHub Email :
set /p email=

echo Please enter github repository link :
set /p url=

REM git global user.name
git config --global user.name %username%
git config --global user.email %email%

REM Clone github repository Change URL here
mkdir repository 
git clone %url% "repository"

REM copy requirements.txt to repository folder
echo copying requirements.txt to repository folder
copy requirements.txt repository

REM go into repo folder
echo changing directory to repository folder
cd repository

REM Create a virtual environment called "venv"
echo creating a virtual environment
python -m venv venv

REM Activate the virtual environment
echo activating virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
echo upgrading pip
python.exe -m pip install --upgrade pip

REM Install any required packages using pip
echo installing requirements
pip install -r requirements.txt

REM Launch VS Code and ensure that it uses the virtual environment
echo opening the repository folder 
code .
