import yaml
import subprocess
import os

def run_segmentation(target_path:str,
                     config_path='./config.yaml'):
    """
    Run segmentation process on images.
    
    Args:
        config_path (str): Path to the configuration YAML file
        
    Returns:
        dict: Dictionary containing the results of the segmentation process,
              including the latest output directory and command execution result
    """
    # Load the configuration file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Read the segment_tool_path and source_path from the configuration
    segment_tool_path = config.get('segment_setting', {}).get('segment_tool_path')
    source_path = config.get('wrap_setting', {}).get('target_folder')
    
    print(f"Segment Tool Path: {segment_tool_path}")
    print(f"Source Path: {source_path}")
    
    # Store original directory to return to it later
    original_dir = os.getcwd()
    
    try:
        # Change the current working directory to segment_tool_path
        os.chdir(segment_tool_path)
        
        # Define the command to execute the demo script
        command = [
            'python',
            'demo.py',
            '--source', target_path,
            '--save-conf',
            '--img-size', '640',
        ]
        
        # Execute the command
        result = subprocess.run(command, text=True, capture_output=True)
        
        print(f"Latest output directory: {target_path}")

    finally:
        # Return to the original directory
        os.chdir(original_dir)

if __name__ == "__main__":
    # Execute the function when the script is run directly
    result = run_segmentation()
    print(f"Segmentation completed successfully: {result['success']}")
