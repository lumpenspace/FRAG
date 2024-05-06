import subprocess


def main() -> None:
    subprocess.run(["sphinx-build", "-b", "html", "docs/", "docs/_build/html"], check=True)


if __name__ == "__main__":
    main()
