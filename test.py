import pkg_resources
import sys
import subprocess

print(f"Python version: {sys.version}")
#installed_packages = [pkg.key for pkg in pkg_resources.working_set]
#print(f"Installed packages: {installed_packages}")

# Read from user-input.txt
with open("C:/Users/erik/Documents/Unreal Projects/MyProject/Content/Python/user-input.txt", "r", encoding="utf-8") as input_file:
    user_text = input_file.read()
    print("model začenja")

# Write to odgovor.txt
with open("C:/Users/erik/Documents/Unreal Projects/MyProject/Content/Python/odgovor.txt", "w", encoding="utf-8") as output_file:
    output_file.write(user_text)
    output_file.write(" odgovor")
    #print("done")


""" import sys
import subprocess

# Path to the desired Python version
python_path = "C:/Users/erik/AppData/Local/Programs/Python/Python312/python.exe"

# Check if the current Python version matches the desired version
if sys.executable != python_path:
    print(f"Switching to Python at: {python_path}")
    subprocess.run([python_path, *sys.argv])  # Re-run the script with the desired Python version
    sys.exit()  # Exit the current script to avoid running it twice

# If already running in the desired Python version, continue execution
print(f"Running in Python version: {sys.version}")

# Read from user-input.txt
with open("C:/Users/erik/Documents/Unreal Projects/MyProject/Content/Python/user-input.txt", "r", encoding="utf-8") as input_file:
    user_text = input_file.read()
    print("model začenja")

# Write to odgovor.txt
with open("C:/Users/erik/Documents/Unreal Projects/MyProject/Content/Python/odgovor.txt", "w", encoding="utf-8") as output_file:
    output_file.write(user_text)
    output_file.write(" odgovor") """