import subprocess
import os
import glob
import shutil
import pytest
import re

# Define the path to the script to be tested
SCRIPT_PATH = "scripts/rfdiffusion_inference.py"

# Fixture to clean up specific output files/dirs created by tests
@pytest.fixture(scope="function")
def cleanup_test_outputs():
    test_prefixes = ["out_test_defaults", "test_prefix_arg", "out_test_num_designs"]
    # This will run before each test that uses it

    # Clean up directories that might be created by Hydra based on output_prefix
    for prefix in test_prefixes:
        # Hydra output dir is often <output_prefix>/hydra_outputs/... or just <output_prefix>/
        # or <output_prefix> if hydra.output_subdir is null
        # The cli_main function sets hydra.run.dir to f"{args.output_prefix}/hydra_outputs/${{hydra.job.name}}"
        # So, we expect a directory named after the prefix.
        dir_to_remove = prefix
        if os.path.exists(dir_to_remove) and os.path.isdir(dir_to_remove):
            try:
                shutil.rmtree(dir_to_remove)
                print(f"Pre-test cleanup: Removed directory {dir_to_remove}")
            except OSError as e:
                print(f"Pre-test cleanup error removing directory {dir_to_remove}: {e}")
        # Also clean any specific files if they somehow exist from a previous failed run
        for f_glob in [f"{prefix}_*.pdb", f"{prefix}_*.trb", f"{prefix}.pdb", f"{prefix}.trb"]:
            for f_path in glob.glob(f_glob):
                if os.path.exists(f_path):
                    try:
                        os.remove(f_path)
                    except OSError as e:
                        print(f"Pre-test cleanup error removing file {f_path}: {e}")


    yield # Test runs here

    # This will run after each test that uses it
    for prefix in test_prefixes:
        dir_to_remove = prefix
        if os.path.exists(dir_to_remove) and os.path.isdir(dir_to_remove):
            try:
                shutil.rmtree(dir_to_remove)
                print(f"Post-test cleanup: Removed directory {dir_to_remove}")
            except OSError as e:
                print(f"Post-test cleanup error removing directory {dir_to_remove}: {e}")
        for f_glob in [f"{prefix}_*.pdb", f"{prefix}_*.trb", f"{prefix}.pdb", f"{prefix}.trb"]:
            for f_path in glob.glob(f_glob):
                if os.path.exists(f_path):
                    try:
                        os.remove(f_path)
                    except OSError as e:
                        print(f"Post-test cleanup error removing file {f_path}: {e}")

    # General cleanup for any default hydra_outputs directory if not caught by prefix logic
    if os.path.exists("hydra_outputs") and os.path.isdir("hydra_outputs"):
         try:
            shutil.rmtree("hydra_outputs")
            print(f"Post-test cleanup: Removed directory hydra_outputs")
         except OSError as e:
            print(f"Post-test cleanup error removing directory hydra_outputs: {e}")


def run_script_and_capture_output(cmd_args, timeout=60):
    """Helper function to run the script and return its output."""
    base_cmd = ["python", SCRIPT_PATH, "--test-dry-run"]
    full_cmd = base_cmd + cmd_args
    try:
        result = subprocess.run(
            full_cmd,
            capture_output=True, text=True, check=False, timeout=timeout
        )
        return result
    except subprocess.TimeoutExpired:
        pytest.fail(f"Script execution timed out ({timeout}s): {' '.join(full_cmd)}")


def assert_config_value_in_output(output, key, expected_value):
    """Asserts that the specific config key/value is printed in the script's output."""
    pattern = re.compile(f"^{key}: {re.escape(expected_value)}$", re.MULTILINE)
    assert pattern.search(output), f"Expected '{key}: {expected_value}' not found in output.\nOutput:\n{output}"


def test_cli_output_prefix_override(cleanup_test_outputs):
    """Test if --output_prefix correctly overrides the hydra config."""
    test_prefix = "test_prefix_arg"
    cmd_args = [
        f"--output_prefix={test_prefix}",
        # No need for inference.num_designs=0 due to --test-dry-run
    ]
    result = run_script_and_capture_output(cmd_args)

    assert_config_value_in_output(result.stdout, "TEST_CONF_OUTPUT_PREFIX", test_prefix)
    assert "TEST_DRY_RUN_ACTIVE" in result.stdout, "Dry run mode was not activated."
    assert "Error" not in result.stderr, f"Script produced an error: {result.stderr}"
    assert "Traceback" not in result.stderr, f"Script crashed with traceback: {result.stderr}"


def test_cli_num_designs_override(cleanup_test_outputs):
    """Test if --num_designs correctly overrides the hydra config."""
    test_num_designs = "3"
    cmd_args = [
        "--num_designs=" + test_num_designs,
        "inference.output_prefix=out_test_num_designs", # Hydra still needs an output dir
    ]
    result = run_script_and_capture_output(cmd_args)

    assert_config_value_in_output(result.stdout, "TEST_CONF_NUM_DESIGNS", test_num_designs)
    assert "TEST_DRY_RUN_ACTIVE" in result.stdout, "Dry run mode was not activated."
    assert "Error" not in result.stderr, f"Script produced an error: {result.stderr}"
    assert "Traceback" not in result.stderr, f"Script crashed with traceback: {result.stderr}"


def test_run_with_default_hydra_config(cleanup_test_outputs):
    """Test running with default hydra config (--config-name=base) and minimal overrides for speed."""
    test_prefix = "out_test_defaults"
    test_num_designs = "0" # Check if this override also works
    cmd_args = [
        "--config-name=base",
        f"inference.num_designs={test_num_designs}",
        f"inference.output_prefix={test_prefix}",
    ]
    result = run_script_and_capture_output(cmd_args)

    assert_config_value_in_output(result.stdout, "TEST_CONF_OUTPUT_PREFIX", test_prefix)
    assert_config_value_in_output(result.stdout, "TEST_CONF_NUM_DESIGNS", test_num_designs)
    assert "TEST_DRY_RUN_ACTIVE" in result.stdout, "Dry run mode was not activated."
    assert "Error" not in result.stderr, f"Script produced an error: {result.stderr}"
    assert "Traceback" not in result.stderr, f"Script crashed with traceback: {result.stderr}"

# Note on cleanup:
# The `hydra.run.dir` is set in `cli_main` to `args.output_prefix + "/hydra_outputs/..."`
# If `args.output_prefix` (from CLI) is e.g. "test_prefix_arg", then hydra's output dir becomes "test_prefix_arg/hydra_outputs".
# The cleanup fixture is designed to remove the top-level "test_prefix_arg" directory.
# If `inference.output_prefix` is not overridden by CLI, it defaults (e.g. from base.yaml, often "out").
# So, `out/hydra_outputs/...` would be created. The cleanup fixture also tries to remove "hydra_outputs" if it exists at root.
# And specific `out_test_num_designs` etc. directories.
