import subprocess
import os
import shutil
import pytest
import re

SCRIPT_PATH = "scripts/rfdiffusion_inference.py"

@pytest.fixture(scope="function")
def cleanup_test_outputs(request):
    """Cleans up any potential output directories created by tests, though less likely with --test-argparse-only."""
    # Define a list of potential output prefixes/directories tests might use
    # This fixture is more of a precaution for this test suite.
    possible_prefixes = ["test_argparse_prefix", "cl_arg_test"]

    yield

    for prefix in possible_prefixes:
        # Remove directories that might have been created by Hydra's default output mechanism
        # (e.g., if a test accidentally runs without --test-argparse-only or --test-dry-run)
        if os.path.exists(prefix) and os.path.isdir(prefix):
            try:
                shutil.rmtree(prefix)
                # print(f"Cleaned up directory: {prefix}")
            except OSError as e:
                print(f"Error removing directory {prefix} during cleanup: {e}")
        # Remove specific files if any test were to create them (unlikely for these tests)
        for f_glob in [f"{prefix}_*.pdb", f"{prefix}_*.trb"]:
            for f_path in glob.glob(f_glob):
                if os.path.exists(f_path):
                    os.remove(f_path)


def run_argparse_test_script(cli_args):
    """Helper to run the script with --test-argparse-only and return stdout."""
    cmd = ["python", SCRIPT_PATH, "--test-argparse-only"] + cli_args
    result = subprocess.run(cmd, capture_output=True, text=True, check=True) # check=True will raise CalledProcessError for non-zero exit
    return result.stdout

def assert_argparse_output(stdout, arg_key, expected_value):
    """Asserts that the expected argparse output line is in stdout."""
    # Ensure expected_value is string for regex matching, as None will be 'None'
    expected_value_str = str(expected_value)
    pattern = re.compile(f"^{arg_key}:{re.escape(expected_value_str)}$", re.MULTILINE)
    assert pattern.search(stdout), f"Expected '{arg_key}:{expected_value_str}' not found in stdout.\nStdout:\n{stdout}"


def test_argparse_output_prefix(cleanup_test_outputs):
    prefix_val = "test_argparse_prefix"
    stdout = run_argparse_test_script(["--output_prefix", prefix_val])
    assert_argparse_output(stdout, "ARGPARSE_OUTPUT_PREFIX", prefix_val)

def test_argparse_num_designs(cleanup_test_outputs):
    num_val = "5"
    stdout = run_argparse_test_script(["--num_designs", num_val])
    assert_argparse_output(stdout, "ARGPARSE_NUM_DESIGNS", num_val) # argparse converts to int, but we print it

def test_argparse_design_startnum(cleanup_test_outputs):
    startnum_val = "10"
    stdout = run_argparse_test_script(["--design_startnum", startnum_val])
    assert_argparse_output(stdout, "ARGPARSE_DESIGN_STARTNUM", startnum_val)

def test_argparse_quiver_path(cleanup_test_outputs):
    quiver_val = "/path/to/my.qv"
    stdout = run_argparse_test_script(["--quiver", quiver_val])
    assert_argparse_output(stdout, "ARGPARSE_QUIVER", quiver_val)

def test_argparse_write_trajectory_true(cleanup_test_outputs):
    stdout = run_argparse_test_script(["--write_trajectory"]) # Action 'store_true'
    assert_argparse_output(stdout, "ARGPARSE_WRITE_TRAJECTORY", True)

def test_argparse_write_trajectory_false(cleanup_test_outputs):
    # For BooleanOptionalAction, absence of flag means default (False if not specified otherwise)
    # Or, it could be `--no-write-trajectory`. Our current setup is `action=argparse.BooleanOptionalAction`
    # which is good, it means it defaults to False if not present.
    # The script prints the default value if the flag is not passed.
    stdout = run_argparse_test_script([])
    assert_argparse_output(stdout, "ARGPARSE_WRITE_TRAJECTORY", False) # Default for BooleanOptionalAction

def test_argparse_deterministic_true(cleanup_test_outputs):
    stdout = run_argparse_test_script(["--deterministic"])
    assert_argparse_output(stdout, "ARGPARSE_DETERMINISTIC", True)

def test_argparse_deterministic_false(cleanup_test_outputs):
    stdout = run_argparse_test_script([])
    assert_argparse_output(stdout, "ARGPARSE_DETERMINISTIC", False)

def test_argparse_cautious_true(cleanup_test_outputs):
    # Current script: parser.add_argument('--cautious', type=lambda x: (str(x).lower() == 'true'), default=True)
    # This means --cautious False should set it to False. Absence should be True.
    # My argparse in cli_main is BooleanOptionalAction, default is False.
    # Let's assume BooleanOptionalAction, which defaults to False if not present.
    stdout = run_argparse_test_script(["--cautious"])
    assert_argparse_output(stdout, "ARGPARSE_CAUTIOUS", True)

def test_argparse_cautious_false(cleanup_test_outputs):
     # The script's argparse for --cautious is BooleanOptionalAction.
     # If the flag is NOT provided, it defaults to False.
     # If --no-cautious is provided, it's False.
     # If --cautious is provided, it's True.
    stdout = run_argparse_test_script(["--no-cautious"]) # Test the --no-<flag> feature
    assert_argparse_output(stdout, "ARGPARSE_CAUTIOUS", False)


def test_argparse_test_dry_run_true(cleanup_test_outputs):
    stdout = run_argparse_test_script(["--test-dry-run"])
    assert_argparse_output(stdout, "ARGPARSE_TEST_DRY_RUN", True)

def test_argparse_all_defaults(cleanup_test_outputs):
    """Test that defaults are correctly reported by argparse layer."""
    stdout = run_argparse_test_script([])
    assert_argparse_output(stdout, "ARGPARSE_CONFIG_NAME", 'base') # Default set in script if None
    assert_argparse_output(stdout, "ARGPARSE_OUTPUT_PREFIX", None) # Default is None for the arg itself
    assert_argparse_output(stdout, "ARGPARSE_NUM_DESIGNS", None)
    assert_argparse_output(stdout, "ARGPARSE_DESIGN_STARTNUM", None)
    assert_argparse_output(stdout, "ARGPARSE_QUIVER", None)
    assert_argparse_output(stdout, "ARGPARSE_WRITE_TRAJECTORY", False)
    assert_argparse_output(stdout, "ARGPARSE_DETERMINISTIC", False)
    assert_argparse_output(stdout, "ARGPARSE_CAUTIOUS", False) # BooleanOptionalAction defaults to False
    assert_argparse_output(stdout, "ARGPARSE_TEST_DRY_RUN", False)
    assert_argparse_output(stdout, "ARGPARSE_UNKNOWN_ARGS", [])

def test_argparse_unknown_args(cleanup_test_outputs):
    """Test that unknown arguments are passed through."""
    unknown_arg = "--some-hydra-override=value"
    stdout = run_argparse_test_script([unknown_arg])
    assert_argparse_output(stdout, "ARGPARSE_UNKNOWN_ARGS", [unknown_arg])

def test_argparse_config_name(cleanup_test_outputs):
    config_name = "custom_config"
    stdout = run_argparse_test_script(["--config-name", config_name])
    assert_argparse_output(stdout, "ARGPARSE_CONFIG_NAME", config_name)

# Note: The `type=lambda x: (str(x).lower() == 'true')` for cautious in the original script
# was replaced by `action=argparse.BooleanOptionalAction`. This action handles boolean flags
# more standardly (--flag for True, --no-flag for False, defaults to False if not specified,
# unless default=True is set in add_argument, which is not the case here).
# The tests for cautious reflect this BooleanOptionalAction behavior.
# The default for args.output_prefix, num_designs etc. in argparse is None if not passed.
# The script later sets args.config_name = 'base' if args.config_name is None.
# The ARGPARSE_ prints reflect the direct output of `vars(args)`.
# The default for `output_prefix` in the *Hydra config* (e.g. base.yaml) might be "out",
# but `args.output_prefix` from `argparse` will be `None` if not given on CLI.
# The test `test_argparse_all_defaults` checks these `None` initial values from `argparse`.

# The `BooleanOptionalAction` for `deterministic`, `cautious`, `write_trajectory`, `test_dry_run`
# means they will be False if not present, True if e.g. `--deterministic` is present.
# The script prints the direct `args.FLAG` value.
# My argparse setup for these boolean flags is:
#   parser.add_argument('--write_trajectory', action=argparse.BooleanOptionalAction, help='Override inference.write_trajectory.')
#   parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, help='Override inference.deterministic.')
#   parser.add_argument('--cautious', action=argparse.BooleanOptionalAction, help='Override inference.cautious.')
#   parser.add_argument('--test-dry-run', action='store_true', help='Exit after printing config for testing purposes.')
# Oh, wait. test-dry-run is store_true. It defaults to False. That's fine.
# The others are BooleanOptionalAction. These also default to False if not specified otherwise in add_argument.
# So, ARGPARSE_WRITE_TRAJECTORY:False is correct for an empty command line.
# Same for ARGPARSE_DETERMINISTIC:False and ARGPARSE_CAUTIOUS:False.
# The test for `test_argparse_cautious_false` using `--no-cautious` is correct for BooleanOptionalAction.
# The test for `test_argparse_cautious_true` using `--cautious` is correct.
# All seems consistent with `BooleanOptionalAction` and `store_true`.
