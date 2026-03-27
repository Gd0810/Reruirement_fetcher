from __future__ import annotations

import ast
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import streamlit as st


COMMON_PACKAGE_NAMES = {
    "bs4": "beautifulsoup4",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
}


@dataclass
class FileDependencyReport:
    path: Path
    dependencies: list[str]
    skipped: list[str]
    error: str | None = None


def choose_folder_with_dialog() -> str | None:
    """Open a native folder chooser when Streamlit runs locally."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory()
    root.destroy()
    return folder or None


def iter_python_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*.py") if path.is_file())


def get_local_module_names(folder: Path, py_files: list[Path]) -> set[str]:
    local_modules: set[str] = set()

    for py_file in py_files:
        try:
            relative_parts = py_file.relative_to(folder).with_suffix("").parts
        except ValueError:
            continue

        if not relative_parts:
            continue

        local_modules.add(relative_parts[-1])

        if py_file.name == "__init__.py":
            local_modules.add(relative_parts[0])

    return local_modules


def normalize_dependency_name(module_name: str) -> str:
    return COMMON_PACKAGE_NAMES.get(module_name, module_name)


def is_third_party_module(module_name: str, local_modules: set[str]) -> bool:
    if not module_name:
        return False

    root_name = module_name.split(".")[0]

    if root_name in local_modules:
        return False

    if root_name in sys.stdlib_module_names:
        return False

    spec = importlib.util.find_spec(root_name)
    if spec is None:
        return True

    origin = spec.origin
    if origin in (None, "built-in", "frozen"):
        return False

    normalized_origin = origin.replace("\\", "/").lower()
    if "site-packages" in normalized_origin or "dist-packages" in normalized_origin:
        return True

    return False


def extract_dependencies(py_file: Path, local_modules: set[str]) -> FileDependencyReport:
    try:
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))
    except UnicodeDecodeError:
        return FileDependencyReport(
            path=py_file,
            dependencies=[],
            skipped=[],
            error="Could not read file as UTF-8.",
        )
    except SyntaxError as exc:
        return FileDependencyReport(
            path=py_file,
            dependencies=[],
            skipped=[],
            error=f"Syntax error on line {exc.lineno}: {exc.msg}",
        )

    found_dependencies: set[str] = set()
    skipped_modules: set[str] = set()

    for node in ast.walk(tree):
        module_name = None

        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split(".")[0]
                if is_third_party_module(module_name, local_modules):
                    found_dependencies.add(normalize_dependency_name(module_name))
                else:
                    skipped_modules.add(module_name)

        elif isinstance(node, ast.ImportFrom):
            if node.level and not node.module:
                continue

            module_name = (node.module or "").split(".")[0]
            if not module_name:
                continue

            if node.level > 0:
                skipped_modules.add(module_name)
                continue

            if is_third_party_module(module_name, local_modules):
                found_dependencies.add(normalize_dependency_name(module_name))
            else:
                skipped_modules.add(module_name)

    return FileDependencyReport(
        path=py_file,
        dependencies=sorted(found_dependencies),
        skipped=sorted(skipped_modules),
    )


def scan_folder(folder_path: str) -> tuple[list[FileDependencyReport], list[str], str | None]:
    folder = Path(folder_path).expanduser().resolve()

    if not folder.exists():
        return [], [], "Selected folder does not exist."

    if not folder.is_dir():
        return [], [], "Selected path is not a folder."

    py_files = iter_python_files(folder)
    if not py_files:
        return [], [], "No Python files were found in that folder."

    local_modules = get_local_module_names(folder, py_files)
    reports = [extract_dependencies(py_file, local_modules) for py_file in py_files]

    all_dependencies = sorted(
        {
            dependency
            for report in reports
            for dependency in report.dependencies
        }
    )
    return reports, all_dependencies, None


def main() -> None:
    st.set_page_config(page_title="Python Dependency Scanner", page_icon="PY", layout="wide")
    st.title("Python Dependency Scanner")
    st.write(
        "Choose a folder, scan all `.py` files inside it, and view the third-party "
        "dependencies used across the project."
    )

    if "selected_folder" not in st.session_state:
        st.session_state.selected_folder = ""

    col1, col2 = st.columns([4, 1])
    with col1:
        folder_path = st.text_input(
            "Folder path",
            value=st.session_state.selected_folder,
            placeholder=r"C:\path\to\your\project",
        )
    with col2:
        st.write("")
        if st.button("Browse"):
            chosen_folder = choose_folder_with_dialog()
            if chosen_folder:
                st.session_state.selected_folder = chosen_folder
                st.rerun()

    scan_clicked = st.button("Scan Dependencies", type="primary")

    if not scan_clicked:
        st.info("Enter a folder path or use Browse, then click Scan Dependencies.")
        return

    reports, all_dependencies, error = scan_folder(folder_path.strip())
    if error:
        st.error(error)
        return

    st.subheader("All Dependencies")
    if all_dependencies:
        st.code("\n".join(all_dependencies), language="text")
    else:
        st.success("No third-party dependencies were found in the Python files.")

    st.subheader("Per File Results")
    for report in reports:
        with st.expander(str(report.path), expanded=False):
            if report.error:
                st.error(report.error)
                continue

            if report.dependencies:
                st.write("Dependencies found:")
                st.code("\n".join(report.dependencies), language="text")
            else:
                st.write("No third-party dependencies found in this file.")

            if report.skipped:
                st.caption(
                    "Ignored standard library or local imports: "
                    + ", ".join(report.skipped)
                )


if __name__ == "__main__":
    main()
