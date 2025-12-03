
import os
os.chdir("./ResNet/")
scripts = [
    "./ResNet/train_model18.py",
    "./ResNet/train_model34.py",
    "./ResNet/train_model18 step 2.py",
    "./ResNet/train_model34 step 2.py"
]

for script in scripts:
    os.system(f'python {script}')