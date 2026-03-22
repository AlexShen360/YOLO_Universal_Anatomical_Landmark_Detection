import sys
import os
sys.path.append('.')

from universal_landmark_detection.model.runner import Runner
import argparse

def test_hip_training_config():
    """Test that hip training configuration works correctly"""

    # Create mock arguments similar to what main.py would receive
    class MockArgs:
        def __init__(self):
            self.config = 'test_hip_config.yaml'
            self.checkpoint = ''
            self.cuda_devices = '0'
            self.model = None
            self.localNet = None
            self.name_list = None
            self.epochs = None
            self.lr = None
            self.weight_decay = None
            self.sigma = None
            self.mix_step = None
            self.use_background_channel = False
            self.run_name = 'test_hip_run'
            self.run_dir = './test_runs'
            self.phase = 'validate'  # Use validate to avoid long training

    try:
        print("Testing Hip training configuration...")

        # Create runner with hip configuration
        args = MockArgs()
        runner = Runner(args)

        print("✓ Runner created successfully")

        # Configure the runner (this calls get_loader, get_model, etc.)
        runner.config()

        print("✓ Runner configured successfully")
        print(f"✓ Dataset name list: {runner.name_list}")
        print(f"✓ Number of datasets: {len(runner.name_list)}")

        # Check if hip is in the name list
        if 'hip' in runner.name_list:
            print("✓ Hip dataset found in name_list")
        else:
            print("✗ Hip dataset NOT found in name_list")
            return False

        # Check dataset configurations
        dataset_opts = runner.opts.dataset
        if 'hip' in dataset_opts:
            hip_config = dataset_opts['hip']
            print(f"✓ Hip dataset configuration found:")
            print(f"  - Prefix: {hip_config['prefix']}")
            print(f"  - Num landmarks: {hip_config['num_landmark']}")
            print(f"  - Size: {hip_config['size']}")
            print(f"  - Sigma: {hip_config['sigma']}")
        else:
            print("✗ Hip dataset configuration NOT found")
            return False

        # Check if hip dataset can be loaded
        if len(runner.validate_dataset_list) > 0:
            hip_dataset = None
            for i, name in enumerate(runner.name_list):
                if name == 'hip':
                    hip_dataset = runner.validate_dataset_list[i]
                    break

            if hip_dataset is not None:
                print(f"✓ Hip dataset loaded successfully")
                print(f"  - Dataset length: {len(hip_dataset)}")
                print(f"  - Sample names: {hip_dataset.indexes[:3] if len(hip_dataset.indexes) > 0 else 'No samples'}")

                # Try to load one sample
                if len(hip_dataset) > 0:
                    sample = hip_dataset[0]
                    print(f"  - Sample input shape: {sample['input'].shape}")
                    print(f"  - Sample GT shape: {sample['gt'].shape}")
                    print(f"  - Sample name: {sample['name']}")
            else:
                print("✗ Hip dataset could not be loaded")
                return False

        print("\n✓ Hip training configuration test PASSED!")
        return True

    except Exception as e:
        print(f"✗ Error testing hip training configuration: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up test run directory
        if os.path.exists('./test_runs'):
            import shutil
            shutil.rmtree('./test_runs')

if __name__ == "__main__":
    test_hip_training_config()
