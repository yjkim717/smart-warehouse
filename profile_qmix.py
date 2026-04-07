import cProfile
import pstats
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # CPU only for simple test

from qmix_tensorflow import NUM_EPISODES
# Override NUM_EPISODES to 10 for profiling
with open('qmix_tensorflow.py', 'r') as f:
    code = f.read()

code = code.replace("NUM_EPISODES = 4000", "NUM_EPISODES = 20")
code = code.replace("EVAL_EPISODES = 300", "EVAL_EPISODES = 1")
code = code.replace("MAX_STEPS = config['env'].get('max_steps', 500)", "MAX_STEPS = 20")

with open('qmix_tensorflow_profile.py', 'w') as f:
    f.write(code)

cProfile.run("import qmix_tensorflow_profile", "stats.prof")
p = pstats.Stats("stats.prof")
p.strip_dirs().sort_stats('tottime').print_stats(30)
