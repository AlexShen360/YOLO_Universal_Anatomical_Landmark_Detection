# Test script to verify the modified evaluation_LLD.py works correctly
import sys
import os

# Add the current directory to the path so we can import evaluation_LLD
sys.path.insert(0, os.getcwd())

try:
    from evaluation_LLD import main
    print("Successfully imported main function from evaluation_LLD")
    
    # Test that main() can be called without arguments
    print("Testing main() function call...")
    
    # This should work now without requiring command line arguments
    # Note: It may fail due to missing files, but it should not fail due to argument parsing
    try:
        main()
        print("main() executed successfully!")
    except Exception as e:
        print(f"main() execution resulted in: {type(e).__name__}: {e}")
        # This is expected if the default paths don't exist, but the important thing
        # is that it doesn't fail due to argument parsing issues
        if "required" in str(e).lower() or "argument" in str(e).lower():
            print("ERROR: Still has argument parsing issues!")
        else:
            print("SUCCESS: No argument parsing issues detected!")
            
except ImportError as e:
    print(f"Failed to import: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")