# Good practices for writing documentation

### Line numbers for code snippets
You can add line numbers to code blocks with the following syntax. Note that you can
specify the starting line number in cases where you insert a specific part of a file.

**Example**

This markdown code

    ```py linenums="5"
    def sum(a, b):
        """Returns the sum of a and b"""
        sum_a_b = a + b
        return sum_a_b
    ```

results like that when rendered. 
```py linenums="5"
def sum(a, b):
    """Returns the sum of a and b"""
    sum_a_b = a + b
    return sum_a_b
```

### Highlight line numbers
You can highlight certain line numbers of your code block with the `hl_lines` keyword.
Note that the numbers for the highlighted lines are relative to your code block, while
the total line numbers can start from any number.

**Example**

This markdown code

    ```py linenums="5", hl_lines="2 3"
    def sum(a, b):
        """Returns the sum of a and b"""
        sum_a_b = a + b
        return sum_a_b
    ```

results like that when rendered. 
```py linenums="5", hl_lines="2 3"
def sum(a, b):
    """Returns the sum of a and b"""
    sum_a_b = a + b
    return sum_a_b
```

### File content placeholders
Inserting code snippets of examples often makes understanding the documentation much 
easier. However, keeping these code snippets in sync with the content of the 
files they are referring to can get exhausting and can be easily forgotten.

To avoid this, please use the following syntax for placing a code snippet in the 
documentation you are writing.

```md
§§§<filename>:<start>:<end>§§§
```

This will replace the original line in the markdown file  with the content of the 
file `<filename>` from line `<start>` to line `<end>`. 
The file in the repository will no be changed, but before building the
docs, a script will create a processed copy of the corresponding markdown file.

**Using a URL instead of a file from the repository**

If you want to link a file that is not present in the umami repo, but you have a URL
to that exact file, you can use the following syntax:

```md
§§§url="<url>":<start>:<end>§§§
```

*Note that if you want to link the content of a file living in another gitlab 
repository, you have to use the link pointing to the **raw** file content.*

**Further examples**

Below you can find different versions for inserting different parts of the file 
`examples/plotting/plot_rocs.py` into your markdown file.

| Placeholder | Result |
|-------------|--------|
|`§§§examples/plotting/plot_rocs.py§§§` | whole file |
|`§§§examples/plotting/plot_rocs.py::§§§` | whole file |
|`§§§examples/plotting/plot_rocs.py:10:20§§§` | from line 10 to line 20 |
|`§§§examples/plotting/plot_rocs.py::10§§§` | from top to line 10 |
|`§§§examples/plotting/plot_rocs.py:10§§§` | from line 10 to bottom |
|`§§§examples/plotting/plot_rocs.py:10:§§§` | from line 10 to bottom |


