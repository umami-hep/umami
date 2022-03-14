### Writing documentation

#### File content placeholders
Inserting code snippets of examples often makes understanding the documentation much 
easier. However, keeping code snippets in the docs in sync with the content of the 
files they are referring to can get exhausting and is easily forgotten.

To avoid this, you can make use of the following syntax for placing a code snippet
in the documentation you are writing.

```md
§§§<filename>:<start>:<end>§§§
```

This will replace the original line in the markdown file  with the content of the 
file `<filename>` from line `<start>` to line `<end>`. 
The file in the repository will no be changed, but before building the
docs, a script will create a processed copy of the corresponding markdown file.

**Further examples**

Below you can find different versions for inserting (parts) of the file 
`examples/plotting/plot_rocs.py` into your markdown file.

| Placeholder | Result |
|-------------|--------|
|`§§§examples/plotting/plot_rocs.py§§§` | whole file |
|`§§§examples/plotting/plot_rocs.py::§§§` | whole file |
|`§§§examples/plotting/plot_rocs.py:10:20§§§` | from line 10 to line 20 |
|`§§§examples/plotting/plot_rocs.py::10§§§` | from top to line 10 |
|`§§§examples/plotting/plot_rocs.py:10§§§` | from line 10 to bottom |
|`§§§examples/plotting/plot_rocs.py:10:§§§` | from line 10 to bottom |