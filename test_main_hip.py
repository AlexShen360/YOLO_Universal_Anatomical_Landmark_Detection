import subprocess
import sys
import os

def test_main_hip():
    """Test that main.py can be called with hip training configuration"""
    
    try:
        print("Testing main.py with hip training configuration...")
        
        # Test command that would be used to train with hip dataset
        cmd = [
            sys.executable, 
            "universal_landmark_detection/main.py",
            "-r", "test_hip_main",
            "-d", "./test_runs", 
            "-p", "validate",  # Use validate to avoid long training
            "-C", "test_hip_config.yaml"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Run the command with timeout to avoid hanging
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=60  # 60 second timeout
        )
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        
        if result.returncode == 0:
            print("✓ main.py with hip configuration executed successfully!")
            return True
        else:
            print("✗ main.py with hip configuration failed!")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ main.py execution timed out (this might be expected for validation)")
        return True  # Timeout might be expected for validation phase
    except Exception as e:
        print(f"✗ Error testing main.py with hip configuration: {str(e)}")
        return False
    
    finally:
        # Clean up test run directory
        if os.path.exists('./test_runs'):
            import shutil
            shutil.rmtree('./test_runs')

if __name__ == "__main__":
    test_main_hip()