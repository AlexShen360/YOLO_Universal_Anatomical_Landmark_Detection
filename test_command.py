import sys
import os
sys.path.append('.')

# Test the specific command parameters
def test_command():
    """Test that the command parameters work with the modified config.yaml"""
    
    try:
        print("Testing command parameters...")
        
        # Import the main module to test argument parsing
        from universal_landmark_detection.main import get_args
        from universal_landmark_detection.model.runner import Runner
        
        # Simulate the command line arguments
        test_args = [
            "main.py",
            "-d", "../runs",
            "-r", "GU2Net_runs", 
            "-p", "validate",  # Use validate instead of train for quick test
            "-m", "gln",
            "-l", "u2net",
            "-n", "hip",
            "-e", "100"
        ]
        
        # Temporarily replace sys.argv
        original_argv = sys.argv
        sys.argv = test_args
        
        try:
            # Test argument parsing
            args = get_args()
            print("✓ Arguments parsed successfully:")
            print(f"  run_dir: {args.run_dir}")
            print(f"  run_name: {args.run_name}")
            print(f"  phase: {args.phase}")
            print(f"  model: {args.model}")
            print(f"  localNet: {args.localNet}")
            print(f"  name_list: {args.name_list}")
            print(f"  epochs: {args.epochs}")
            
            # Test runner initialization
            runner = Runner(args)
            print("✓ Runner created successfully")
            
            # Test configuration loading
            runner.get_opts()
            print("✓ Configuration loaded successfully")
            print(f"  Dataset name_list: {runner.opts.dataset.name_list}")
            print(f"  Model: {runner.opts.model}")
            print(f"  GLN localNet: {runner.opts.gln.localNet}")
            print(f"  Epochs: {runner.opts.epochs}")
            
            # Check if hip configuration exists
            if 'hip' in runner.opts.dataset:
                hip_config = runner.opts.dataset.hip
                print("✓ Hip dataset configuration found:")
                print(f"  Prefix: {hip_config.prefix}")
                print(f"  Num landmarks: {hip_config.num_landmark}")
                print(f"  Size: {hip_config.size}")
            else:
                print("✗ Hip dataset configuration not found")
                return False
                
            print("\n✓ Command test PASSED! The command should work correctly.")
            return True
            
        finally:
            # Restore original sys.argv
            sys.argv = original_argv
            
    except Exception as e:
        print(f"✗ Error testing command: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_command()