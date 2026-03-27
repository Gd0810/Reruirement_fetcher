from __future__ import annotations

import ast
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st
import tomllib


COMMON_PACKAGE_NAMES = {
    "bs4": "beautifulsoup4",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "dotenv": "python-dotenv",
    "jwt": "PyJWT",
    "qdrant_client": "qdrant-client",
    "rest_framework": "djangorestframework",
    "sentence_transformers": "sentence-transformers",
    "sklearn": "scikit-learn",
    "langchain_text_splitters": "langchain-text-splitters",
    "yaml": "PyYAML",
}

IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "env",
    "site-packages",
    "venv",
}


@dataclass
class FileDependencyReport:
    path: Path
    imports: list[str]
    package_names: list[str]
    dynamic_imports: list[str]
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
    python_files: list[Path] = []

    for path in folder.rglob("*.py"):
        if not path.is_file():
            continue

        if any(part.lower() in IGNORED_DIRECTORY_NAMES for part in path.parts):
            continue

        python_files.append(path)

    return sorted(python_files)


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


def strip_version_specifier(requirement: str) -> str:
    cleaned = requirement.strip()
    if not cleaned or cleaned.startswith("#"):
        return ""

    if ";" in cleaned:
        cleaned = cleaned.split(";", 1)[0].strip()

    if "[" in cleaned:
        cleaned = cleaned.split("[", 1)[0].strip()

    for separator in ("==", ">=", "<=", "~=", "!=", ">", "<"):
        if separator in cleaned:
            cleaned = cleaned.split(separator, 1)[0].strip()
            break

    return cleaned


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


def get_literal_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def extract_dependencies(py_file: Path, local_modules: set[str]) -> FileDependencyReport:
    try:
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))
    except UnicodeDecodeError:
        return FileDependencyReport(
            path=py_file,
            imports=[],
            package_names=[],
            dynamic_imports=[],
            skipped=[],
            error="Could not read file as UTF-8.",
        )
    except SyntaxError as exc:
        return FileDependencyReport(
            path=py_file,
            imports=[],
            package_names=[],
            dynamic_imports=[],
            skipped=[],
            error=f"Syntax error on line {exc.lineno}: {exc.msg}",
        )

    found_imports: set[str] = set()
    found_packages: set[str] = set()
    found_dynamic_imports: set[str] = set()
    skipped_modules: set[str] = set()

    for node in ast.walk(tree):
        module_name = None

        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split(".")[0]
                if is_third_party_module(module_name, local_modules):
                    found_imports.add(module_name)
                    found_packages.add(normalize_dependency_name(module_name))
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
                found_imports.add(module_name)
                found_packages.add(normalize_dependency_name(module_name))
            else:
                skipped_modules.add(module_name)

        elif isinstance(node, ast.Call):
            module_name = None

            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "__import__"
                and node.args
            ):
                module_name = get_literal_string(node.args[0])

            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "importlib"
                and node.func.attr == "import_module"
                and node.args
            ):
                module_name = get_literal_string(node.args[0])

            if module_name:
                module_name = module_name.split(".")[0]
                if is_third_party_module(module_name, local_modules):
                    found_dynamic_imports.add(module_name)
                    found_packages.add(normalize_dependency_name(module_name))
                else:
                    skipped_modules.add(module_name)

    return FileDependencyReport(
        path=py_file,
        imports=sorted(found_imports),
        package_names=sorted(found_packages),
        dynamic_imports=sorted(found_dynamic_imports),
        skipped=sorted(skipped_modules),
    )


def read_requirements_file(requirements_path: Path) -> list[str]:
    dependencies: list[str] = []
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        stripped = strip_version_specifier(line)
        if stripped:
            dependencies.append(stripped)
    return sorted(set(dependencies), key=str.lower)


def extract_pyproject_dependencies(pyproject_data: dict[str, Any]) -> list[str]:
    dependencies: set[str] = set()

    project = pyproject_data.get("project", {})
    for dependency in project.get("dependencies", []):
        cleaned = strip_version_specifier(str(dependency))
        if cleaned:
            dependencies.add(cleaned)

    optional_dependencies = project.get("optional-dependencies", {})
    for group_dependencies in optional_dependencies.values():
        for dependency in group_dependencies:
            cleaned = strip_version_specifier(str(dependency))
            if cleaned:
                dependencies.add(cleaned)

    poetry = pyproject_data.get("tool", {}).get("poetry", {})
    for name, value in poetry.get("dependencies", {}).items():
        if name.lower() == "python":
            continue
        if isinstance(value, str):
            dependencies.add(name)
        elif isinstance(value, dict):
            dependencies.add(name)

    return sorted(dependencies, key=str.lower)


def read_pyproject_dependencies(pyproject_path: Path) -> list[str]:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    return extract_pyproject_dependencies(data)


def collect_declared_dependencies(folder: Path) -> tuple[dict[str, list[str]], list[str]]:
    declared_by_file: dict[str, list[str]] = {}

    for requirements_path in sorted(folder.rglob("requirements*.txt")):
        if any(part.lower() in IGNORED_DIRECTORY_NAMES for part in requirements_path.parts):
            continue
        dependencies = read_requirements_file(requirements_path)
        if dependencies:
            declared_by_file[str(requirements_path)] = dependencies

    for pyproject_path in sorted(folder.rglob("pyproject.toml")):
        if any(part.lower() in IGNORED_DIRECTORY_NAMES for part in pyproject_path.parts):
            continue
        dependencies = read_pyproject_dependencies(pyproject_path)
        if dependencies:
            declared_by_file[str(pyproject_path)] = dependencies

    combined = sorted(
        {
            dependency
            for dependencies in declared_by_file.values()
            for dependency in dependencies
        },
        key=str.lower,
    )
    return declared_by_file, combined


def scan_folder(
    folder_path: str,
) -> tuple[list[FileDependencyReport], list[str], list[str], dict[str, list[str]], list[str], str | None]:
    folder = Path(folder_path).expanduser().resolve()

    if not folder.exists():
        return [], [], [], {}, [], "Selected folder does not exist."

    if not folder.is_dir():
        return [], [], [], {}, [], "Selected path is not a folder."

    py_files = iter_python_files(folder)
    if not py_files:
        return [], [], [], {}, [], "No Python files were found in that folder."

    local_modules = get_local_module_names(folder, py_files)
    reports = [extract_dependencies(py_file, local_modules) for py_file in py_files]

    all_imports = sorted(
        {
            dependency
            for report in reports
            for dependency in report.imports
        },
        key=str.lower,
    )
    all_package_names = sorted(
        {
            dependency
            for report in reports
            for dependency in report.package_names
        }
    ,
        key=str.lower,
    )
    declared_by_file, declared_dependencies = collect_declared_dependencies(folder)
    return (
        reports,
        all_imports,
        all_package_names,
        declared_by_file,
        declared_dependencies,
        None,
    )


def main() -> None:
    st.set_page_config(page_title="Python Dependency Scanner", page_icon="PY", layout="wide")
    st.title("Python Dependency Scanner")
    st.write(
        "Choose a folder, scan all `.py` files inside it, and view the third-party "
        "dependencies used across the project."
    )
    st.caption(
        "Code scanning finds direct imports used by the project. Exact installed packages "
        "come from files like `requirements.txt`, `pyproject.toml`, or the active environment."
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

    reports, all_imports, all_package_names, declared_by_file, declared_dependencies, error = scan_folder(
        folder_path.strip()
    )
    if error:
        st.error(error)
        return

    st.subheader("Direct Imports Found In Code")
    if all_imports:
        st.code("\n".join(all_imports), language="text")
    else:
        st.success("No third-party imports were found in the Python files.")

    st.subheader("Estimated Pip Package Names")
    if all_package_names:
        st.code("\n".join(all_package_names), language="text")
    else:
        st.info("No estimated package names were detected from imports.")

    st.subheader("Declared Dependencies From Project Files")
    if declared_dependencies:
        st.code("\n".join(declared_dependencies), language="text")
    else:
        st.info("No `requirements*.txt` or `pyproject.toml` dependencies were found.")

    if declared_by_file:
        st.write("Dependencies read from project files:")
        for file_path, dependencies in declared_by_file.items():
            with st.expander(file_path, expanded=False):
                st.code("\n".join(dependencies), language="text")

    st.subheader("Per File Results")
    for report in reports:
        with st.expander(str(report.path), expanded=False):
            if report.error:
                st.error(report.error)
                continue

            if report.imports:
                st.write("Direct imports found:")
                st.code("\n".join(report.imports), language="text")
                if report.dynamic_imports:
                    st.write("Dynamic imports found:")
                    st.code("\n".join(report.dynamic_imports), language="text")
                st.write("Estimated package names:")
                st.code("\n".join(report.package_names), language="text")
            else:
                st.write("No third-party dependencies found in this file.")

            if report.skipped:
                st.caption(
                    "Ignored standard library or local imports: "
                    + ", ".join(report.skipped)
                )


if __name__ == "__main__":
    main()
