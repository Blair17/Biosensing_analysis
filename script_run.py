import subprocess

def run_script(script_path):
    subprocess.run(["python", script_path])

run_script('Image_analysis.py')
run_script('Concentration_plot.py')

# run_script('Glucose_data_extraction.py')
# run_script('Glucose_plotting.py')