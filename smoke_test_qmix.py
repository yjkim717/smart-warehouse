import json

print("1. Loading Jupyter Notebook...")
with open("qmix_tensorflow.ipynb") as f:
    notebook = json.load(f)

code = ""
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        code += "".join(cell["source"]) + "\n"

print("2. Modifying Hyperparameters for Debugging...")
code = code.replace("NUM_EPISODES = 4000", "NUM_EPISODES = 2")
code = code.replace("EVAL_EPISODES = 300", "EVAL_EPISODES = 2")
code = code.replace("TARGET_UPDATE_INTERVAL = 200", "TARGET_UPDATE_INTERVAL = 1")

# Disable persistent blocking plots
code = code.replace("plt.show()", "import os\nos.makedirs('results/plots', exist_ok=True)\nplt.savefig('results/plots/qmix_tf_test_panel.png')\nplt.close()")

print("3. Executing Neural Network & Environment Flow...")
try:
    exec(code, globals())
    print("-------------------------")
    print("✅ SMOKE TEST PASSED: No configuration or dimensional limits found.")
except Exception as e:
    print("-------------------------")
    print("❌ SMOKE TEST FAILED.")
    raise e
