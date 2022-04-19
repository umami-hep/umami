"""Script to replace placeholders in .md files with the content of the file that is
specified in the placeholder."""
import argparse
import os
import re
from collections import Counter
from glob import glob
from shutil import copyfile
from subprocess import run


def get_parser():
    """
    Argument parser for example code replacer

    Returns
    -------
    args: parse_args

    Raises
    ------
    ValueError
        If output file is specified but multiple input files are given.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Script which allow to replace placeholders inside a markdown "
            "file with the actual content of files located in your repo.\n"
            "Such placeholders have to be of the form §§§<filename>:<start>:<end>§§§. "
            " If start(end) is empty, the first(last) line is used."
            "Indentation of the pasted content is the same as the indentation of"
            "the placeholder. Valid examples are: "
            "§§§file.py§§§, §§§file.py:10§§§, §§§file.py:10:20§§§"
        )
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help=(
            "Name of the input file(s). Should be markdown files. Wildcards are "
            "supported"
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        default=None,
        help=(
            "Name of the output file (.md file). If none is provided, the input file "
            "will be overwritten and a copy of the original file is saved with a .bkp "
            "ending. Only makes sense if `input` is a single file."
        ),
    )

    parser.add_argument(
        "-n",
        "--no_backup",
        action="store_true",
        default=False,
        help="Do not save a backup of the original file.",
    )

    parser.add_argument(
        "-e",
        "--exclude",
        action="append",
        default=None,
        help="Exclude this file.",
        type=str,
    )

    args = parser.parse_args()

    print(args.exclude)

    # Translate wildcard and remove excluded files
    args.input = glob(args.input, recursive=True)
    if args.exclude is not None:
        print(f"\x1b[1;32;40mExcluding the following files: {args.exclude}\x1b[0m")
        args.input = [
            filename for filename in args.input if filename not in args.exclude
        ]

    if len(args.input) != 1 and args.output is not None:
        raise ValueError(
            "You specified an output file but more than one input was given."
            "This option is only supported for a single input file."
        )

    return args


def line_contains_placeholder(line):
    """Helper function to check if a line contains a valid placeholder

    Parameters
    ----------
    line : str
        Line which is checked if a valid placeholder is in there

    Returns
    -------
    bool
        True if all criteria are satisfied, otherwise false
    """

    char_counter = Counter(line)
    if char_counter["§"] == 6:
        return True
    return False


def replace_placeholder_with_file_content(
    input_file: str, output_file: str = None, no_backup: bool = False
):
    """Function to replace placeholders of the form "§§§<filename>:<start>:<end>§§§"
    with the actual content of the file <filename> from line <start> to line <end>.

    Parameters
    ----------
    input_file : str
        Filename of the markdown file which is searched for placeholders of the form
        §§§<filename>:<start>:<end>§§§.
    output_file : str, optional
        Filename of the output file (with the placeholders replaced). If not specified,
        the output file will overwrite the input file and a copy of the original file
        will be saved. By default None
    no_backup : bool, optional
        Option to overwrite original file without saving a backup file, by default False

    Raises
    ------
    ValueError
        If placeholder contains invalid number of colons. Valid numbers are 0, 1, 2
    FileNotFoundError
        If the file specified in a placeholder does not exist.
    """

    print(f"{90 * '-'}\nProcessing {input_file}")
    # Check if any placeholder is in the file. If not, stop here already
    with open(input_file, "r") as original_file:
        if "§§§" not in original_file.read():
            print(f"File {input_file} does not contain any placeholders. Skipped.")
            return

    # Save backup of input file if no output filename is specified
    if output_file is None or output_file == input_file:
        if no_backup is False:
            copyfile(input_file, f"{input_file}.bkp")
            print(
                "Input filename is equal to output filename or no output filename "
                f"specified. Saving backup of input file as {input_file}.bkp"
            )
        output_file = input_file

    output_file_content = ""
    replaced_placeholders = []

    with open(input_file, "r") as original_file:
        # Loop over lines in input file and search for lines containing "§§§"
        for original_line in original_file:
            if line_contains_placeholder(original_line):
                # Extract filename, start line and end line fro placeholder which
                # has to be specified like §§§<filename>:<start>:<end>§§§
                placeholder = original_line.split("§§§")[1]

                # Check if a url was specified. If yes, download the file
                if placeholder.startswith("url="):
                    url_search = re.search('url="(.*)"', placeholder)
                    url = url_search.group(1)
                    os.makedirs("downloads", exist_ok=True)
                    tmp_filename = f"downloads/{url.split('/')[-1]}"
                    print(f"Downloading file {url} -> {tmp_filename}")
                    run(
                        f"wget {url} -O {tmp_filename}",
                        shell=True,
                        check=True,
                    )
                    # replacement_file = tmp_filename
                    placeholder = placeholder.replace(f'url="{url}"', tmp_filename)

                # Check how many colons are in the placeholder
                # Translate to python index + convert to start=0, end=-1 in case
                # where no number is specified
                char_counter = Counter(placeholder)
                if char_counter[":"] == 0:
                    replacement_file = placeholder
                    start, end = 0, None
                elif char_counter[":"] == 1:
                    replacement_file, start = placeholder.split(":")
                    start = 0 if start == "" else int(start) - 1
                    end = None
                elif char_counter[":"] == 2:
                    replacement_file, start, end = placeholder.split(":")
                    start = 0 if start == "" else int(start) - 1
                    end = None if end == "" else int(end)
                else:
                    raise ValueError(
                        f"Placeholder {placeholder} contains more than 2 colons. "
                        "Supported number of colons are 0, 1 and 2."
                    )

                # Extract indentation for replacement lines
                indentation = original_line.split("§§§")[0]

                try:
                    with open(replacement_file, "r") as rep_content:
                        replacement = ""
                        # Add all selected lines from the file, but always add the
                        # indentation of the placeholder
                        for line in rep_content.readlines()[start:end]:
                            replacement += f"{indentation}{line}"
                        replaced_placeholders.append(replacement_file)
                    # Add to md file
                    output_file_content += replacement
                except FileNotFoundError as err:
                    raise FileNotFoundError(
                        "\x1b[1;33;40m"
                        f"file: {input_file}, invalid placeholder: {original_line}"
                        "\x1b[0m"
                    ) from err
            else:
                output_file_content += original_line

    with open(output_file, "w") as md_file_new:
        md_file_new.write(output_file_content)
    print("SUMMARY:")
    print(f"Replaced placeholders: {replaced_placeholders}")


def main():
    """Main function that is called when executing the script."""
    args = get_parser()
    print(f"Replacing placeholders in the following files: {args.input}")

    # Process each input file with the replacement function
    for input_file in args.input:
        replace_placeholder_with_file_content(
            input_file=input_file,
            output_file=args.output if len(args.input) == 1 else None,
            no_backup=args.no_backup,
        )


if __name__ == "__main__":
    main()
