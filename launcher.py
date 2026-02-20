import subprocess

print("Launching script with specific Python version...")  
# Path to the desired Python version
python_path = "C:/Users/erik/AppData/Local/Programs/Python/Python312/python.exe"
# Command to run the script with the desired Python version
subprocess.run([python_path, "test.py"])
