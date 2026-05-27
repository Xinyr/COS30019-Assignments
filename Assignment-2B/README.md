# Group Members:
Frederick Bartholomew Dao Xiang SII, 104397146

Izzat Dasuki Bin Mohamad Hamzah, 104397971


# Usage
1. Navigate to `./Assignment-2B/` in CLI
2. If needed, create the environment first
3. Run the commands below

Environment setup:
1. Create environment: ```conda env create -f environment.yml```
2. Activate environment: ```conda activate a2b-gru```

Current supported commands:
1. Open GUI: ```.\run_gui.bat```
2. Run route demo: ```.\run_route.bat```
3. Run model comparison: ```.\run_models.bat```
4. Open GUI manually: ```C:\Users\User\anaconda3\envs\a2b-gru\python.exe launch_gui.py```
5. Run route system manually: ```C:\Users\User\anaconda3\envs\a2b-gru\python.exe main.py --origin 4335 --destination 3217 --date 10/15/2006 --time 08:00 --model random_forest --search-method astar --top-k 5```
6. Run model evaluation manually: ```C:\Users\User\anaconda3\envs\a2b-gru\python.exe evaluate_models.py --loc-index 1 --time-index 0 --epochs 60```
7. Run tests: ```C:\Users\User\anaconda3\envs\a2b-gru\python.exe -m unittest discover -s tests -v```

Current ml models:
1. Random Forest
2. LSTM
3. GRU

Note:
1. Model output is number of vehicles per 15 minutes
2. The system converts that into travel time for route search
