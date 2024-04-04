import subprocess

def main():
    # This command builds the Sphinx documentation in HTML format
    subprocess.run("make html", check=True)

if __name__ == "__main__":
    main()