"""Extract and test code examples from README.md using doctest format.

This script looks for code blocks containing >>> prompts and tests them
like standard doctests, verifying both execution and output.
"""

import argparse
import re
import sys
from contextlib import redirect_stdout
from io import StringIO

from rich import print


def extract_python_examples(readme_path):
    """Extract Python code blocks from README.

    Args:
        readme_path: Path to the README.md file.

    Yields:
        Tuples of (line_number, code_block).
    """
    with open(readme_path, "r") as f:
        content = f.read()

    # Find all Python code blocks
    pattern = r"```python\n(.*?)\n```"
    matches = re.finditer(pattern, content, re.DOTALL)

    for match in matches:
        # Find the line number
        line_num = content[: match.start()].count("\n") + 1
        code = match.group(1)
        yield line_num, code


def parse_doctest_block(code):
    """Parse a doctest-style code block.

    Args:
        code: Code block text.

    Returns:
        List of (input_lines, expected_output) tuples, or None if not a doctest block.
    """
    lines = code.split("\n")

    # Check if this block contains any >>> prompts
    if not any(line.strip().startswith(">>>") for line in lines):
        return None

    examples = []
    current_input = []
    current_output = []
    in_example = False

    for line in lines:
        stripped = line.strip()

        # Skip blank lines and comments
        if not stripped or stripped.startswith("#"):
            continue

        # Start of a new example
        if stripped.startswith(">>>"):
            # Save previous example if exists
            if current_input:
                examples.append((current_input, current_output))
                current_input = []
                current_output = []

            # Add this line to current input
            current_input.append(stripped[4:])  # Remove '>>> '
            in_example = True

        # Continuation line
        elif stripped.startswith("..."):
            if in_example:
                current_input.append(stripped[4:])  # Remove '... '

        # Expected output line
        else:
            if in_example:
                current_output.append(stripped)

    # Don't forget the last example
    if current_input:
        examples.append((current_input, current_output))

    return examples if examples else None


def normalize_output(output):
    """Normalize output for comparison.

    Args:
        output: Output string.

    Returns:
        Normalized output string.
    """
    # Remove extra whitespace
    output = " ".join(output.split())
    return output


def test_doctest_example(input_lines, expected_output, namespace):
    """Test a single doctest example.

    Args:
        input_lines: List of input code lines.
        expected_output: List of expected output lines.
        namespace: Namespace to execute code in.

    Returns:
        Tuple of (success, actual_output, error_message).
    """
    code = "\n".join(input_lines)

    # Capture output
    stdout = StringIO()
    actual_output = None
    error_msg = None

    try:
        with redirect_stdout(stdout):
            # Execute the code
            result = eval(code, namespace)

            # If eval worked and returned something, that's the output
            if result is not None:
                actual_output = repr(result)
            else:
                # Otherwise, check if anything was printed
                actual_output = stdout.getvalue().strip()

    except SyntaxError:
        # Not an expression, try exec
        try:
            stdout = StringIO()
            with redirect_stdout(stdout):
                exec(code, namespace)
                actual_output = stdout.getvalue().strip()
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            return False, None, error_msg

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        return False, None, error_msg

    # If no expected output, just check it ran without error
    if not expected_output:
        return True, actual_output, None

    # Compare expected vs actual
    expected_str = " ".join(expected_output)
    expected_normalized = normalize_output(expected_str)
    actual_normalized = normalize_output(actual_output)

    if expected_normalized == actual_normalized:
        return True, actual_output, None
    else:
        error_msg = f"Expected: {expected_str}\n         Got: {actual_output}"
        return False, actual_output, error_msg


def test_code_block(code, line_num, namespace):
    """Test a code block containing doctest examples.

    Args:
        code: The code block text.
        line_num: Line number in README for error reporting.

    Returns:
        Tuple of (passed, failed, error_details).
    """
    examples = parse_doctest_block(code)

    if examples is None:
        return 0, 0, []  # Not a doctest block, skip silently

    passed = 0
    failed = 0
    errors = []

    for i, (input_lines, expected_output) in enumerate(examples, 1):
        success, actual, error_msg = test_doctest_example(
            input_lines, expected_output, namespace
        )

        if success:
            passed += 1
        else:
            failed += 1
            errors.append(
                {"example_num": i, "code": "\n".join(input_lines), "error": error_msg}
            )

    return passed, failed, errors


def test_readme(readme_path="README.md", fail_fast=False):
    """Test all doctest examples in README.

    Args:
        readme_path: Path to README.md.
        fail_fast: If True, stop at first failure.

    Returns:
        True if all tests passed, False otherwise.
    """
    # Shared REPL namespace for the entire README
    import json

    import numpy as np

    from varuintarray import VarUIntArray, packbits, unpackbits

    namespace = {
        "__name__": "__doctest__",  # nice to have
        "np": np,
        "VarUIntArray": VarUIntArray,
        "packbits": packbits,
        "unpackbits": unpackbits,
        "json": json,
    }

    total_passed = 0
    total_failed = 0
    blocks_tested = 0
    blocks_skipped = 0

    print(f"Testing doctest examples from {readme_path}...")
    print("-" * 60)

    for line_num, code in extract_python_examples(readme_path):
        passed, failed, errors = test_code_block(code, line_num, namespace)

        if passed == 0 and failed == 0:
            # Not a doctest block
            blocks_skipped += 1
            continue

        blocks_tested += 1
        total_passed += passed
        total_failed += failed

        if failed == 0:
            print(
                f"[green]✓[/green] Block at line {line_num}: {passed} example(s) passed"
            )
        else:
            print(
                f"[red]✗[/red] Block at line {line_num}: {passed} passed, {failed} failed"
            )
            for error in errors:
                print(f"\n  Example {error['example_num']}:")
                print(f"    Code: {error['code']}")
                print(f"    {error['error']}")

            if fail_fast:
                print("\n[FAIL FAST] Stopping at first failure")
                print("\n" + "=" * 60)
                print(f"Blocks: {blocks_tested} tested, {blocks_skipped} skipped")
                print(f"Examples: {total_passed} passed, {total_failed} failed")
                print("=" * 60)
                return False

    print("\n" + "=" * 60)
    print(f"Blocks: {blocks_tested} tested, {blocks_skipped} skipped")

    if total_failed == 0:
        print(f"Examples: {total_passed} passed, {total_failed} failed")
        print("=" * 60)
    else:
        print(f"Examples: {total_passed} passed, {total_failed} failed")
        print("=" * 60)

    return total_failed == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test doctest examples in README.md")
    parser.add_argument(
        "readme",
        nargs="?",
        default="README.md",
        help="Path to README file (default: README.md)",
    )
    parser.add_argument(
        "-x", "--fail-fast", action="store_true", help="Stop at first failure"
    )

    args = parser.parse_args()

    success = test_readme(args.readme, fail_fast=args.fail_fast)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
