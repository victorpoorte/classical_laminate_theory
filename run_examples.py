import subprocess
import pathlib
import sys

EXAMPLES_DIR = pathlib.Path(__file__).parent / "examples"

def module_path_from_main(main_file: pathlib.Path) -> str:
    # Convert examples/foo/bar/main.py → examples.foo.bar.main
    rel_path = main_file.relative_to(EXAMPLES_DIR.parent).with_suffix("")
    return rel_path.as_posix().replace('/', '.')

def run_main(main_file: pathlib.Path):
    module_path = module_path_from_main(main_file)
    print(f"🔧 Running {module_path} ...")

    try:
        subprocess.run(
            [sys.executable, "-m", module_path],
            check=True
        )
        print(f"✅ {module_path} completed\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run {module_path}: {e}\n")

def run_all_mains():
    main_files = list(EXAMPLES_DIR.rglob("main.py"))
    print(f"📁 Found {len(main_files)} example main files.\n")

    for main_file in main_files:
        run_main(main_file)

if __name__ == "__main__":
    run_all_mains()
