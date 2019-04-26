# instruction

    0. Prequsite

        - Tensorflow
        - gym
        - baselines
        - pytest

    1. Install baseline 

        1.1 Download and install
```
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
```
        1.2 Replace the baselines/baselines/run.py with run.py in this folder

    2. Install the stock environment to gym
    
        2.1 Go to the directory of gym library (e.g. /home/user/anaconda3/envs/tensorflow/lib/python3.6/site-packages/gym)
            2.1.1 Please note you need to change the hard-coded directories 
                - in stock/stock_env.py 
                    - change data path and the directory to save graph
                - in stock/stock_testenv.py
                    - change data path and the directory to save graph
        2.2 Copy folder stock/ from this folder to gym/env/
        2.3 Replace __init__.py in the gym/env/ with __init__.py in this folder

    3. Install pytest `pip install pytest`

    4. You should be able to run the test with `python -m baselines.run --alg=ddpg --env=Stock-v0 --network=mlp --num_timesteps=1e4 --play`
