import json

with open('qmix_tensorflow.py', 'r') as f:
    py_code = f.read()

eval_split = py_code.split("# ==========================================\n# Evaluation vs MAPPO & Random Baseline\n# ==========================================\n")
part0 = eval_split[0]

vis_split = eval_split[1].split("# ==========================================\n# Tri-Model 6-Panel Visualization Dashboard\n# ==========================================\n")

part1 = "# ==========================================\n# Evaluation vs MAPPO & Random Baseline\n# ==========================================\n" + vis_split[0]
part2 = "# ==========================================\n# Tri-Model 6-Panel Visualization Dashboard\n# ==========================================\n" + vis_split[1]

with open('qmix_tensorflow.ipynb', 'r') as f:
    nb = json.load(f)

def to_lines(text):
    lines = text.split('\n')
    return [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

# Assuming the first 3 cells are training, evaluation, validation
nb['cells'][0]['source'] = to_lines(part0)
nb['cells'][1]['source'] = to_lines(part1)
nb['cells'][2]['source'] = to_lines(part2)

with open('qmix_tensorflow.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

