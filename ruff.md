*Ruff* is an extremely fast Python linter and code formatter

```sh
ruff check                  # Lint all files in the current directory.
ruff check --fix            # Lint all files in the current directory, and fix any fixable errors.
ruff check --watch          # Lint all files in the current directory, and re-lint on change.
```

I have also configured **pre-commit** as well as **Continuous Integration(CI)** using Github actions.