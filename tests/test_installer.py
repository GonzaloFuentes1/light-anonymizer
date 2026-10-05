"""The Windows installer: packaging/installer.iss and scripts/build_installer.py.

The checks of the script's content and of build_installer.py run everywhere. With Inno Setup 6
installed (Windows only), a tiny fake application is compiled, installed silently into a temporary
folder, blocked by the running-app mutex, upgraded and uninstalled. That installation has its own
AppId, mutex and shortcut name, and uses /NOICONS, so it never touches a real installation.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from anonymizer.api import server

ROOT = Path(__file__).resolve().parents[1]
ISS = ROOT / "packaging" / "installer.iss"
SCRIPT = ISS.read_text(encoding="utf-8-sig")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


builder = load("build_installer", ROOT / "scripts" / "build_installer.py")
launcher = load("launcher", ROOT / "packaging" / "launcher.py")


def section(name: str) -> list[str]:
    """The lines of one [Section], without comments, preprocessor lines or blank lines."""
    lines, current = [], None
    for raw in SCRIPT.splitlines():
        line = raw.strip()
        if re.fullmatch(r"\[\w+\]", line):
            current = line[1:-1]
        elif current == name and line and not line.startswith((";", "#", "//")):
            lines.append(line)
    return lines


def directives(name: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in section(name) if "=" in line)


def entries(name: str) -> list[dict[str, str]]:
    """[Files]-style entries: 'Key: "value"; Key: value' -> dict."""
    found = []
    for line in section(name):
        pairs = re.findall(r'(\w+):\s*("[^"]*"|[^;]*)', line)
        found.append({key: value.strip().strip('"') for key, value in pairs})
    return found


def defines() -> dict[str, str]:
    return dict(re.findall(r'#define\s+(\w+)\s+"([^"]*)"', SCRIPT))


# --- The script ------------------------------------------------------------------------------------


def test_installs_per_user_without_administrator_in_a_short_path():
    setup = directives("Setup")
    assert setup["PrivilegesRequired"] == "lowest"
    assert "PrivilegesRequiredOverridesAllowed" not in setup  # no "install for all users" option
    assert setup["DefaultDirName"] == r"{localappdata}\Programs\LightAnonymizer"
    assert setup["DisableDirPage"] == "yes"
    assert setup["ArchitecturesInstallIn64BitMode"] == "x64compatible"


def test_installer_speaks_spanish():
    languages = entries("Languages")
    assert [entry["MessagesFile"] for entry in languages] == [r"compiler:Languages\Spanish.isl"]
    assert directives("Setup")["ShowLanguageDialog"] == "no"
    texts = [entry["Description"] for entry in entries("Run")]
    assert "Abrir el Anonimizador" in texts


def test_version_comes_from_the_build():
    setup = directives("Setup")
    assert setup["AppVersion"] == "{#AppVersion}"
    assert setup["VersionInfoVersion"] == "{#AppNumericVersion}"
    assert setup["OutputBaseFilename"] == "LightAnonymizer-{#AppVersion}-setup"
    assert builder.installer_name("0.1.0") + ".exe" == "LightAnonymizer-0.1.0-setup.exe"
    statements = [line for line in SCRIPT.splitlines() if not line.lstrip().startswith(";")]
    assert not re.search(r"\b\d+\.\d+\.\d+\b", "\n".join(statements))  # no version written in the script


def test_license_page_shows_the_agpl_without_forcing_acceptance():
    setup = directives("Setup")
    license_file = (ISS.parent / setup["InfoBeforeFile"]).resolve()
    assert license_file == (ROOT / "LICENSE").resolve()
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in license_file.read_text(encoding="utf-8")
    assert "LicenseFile" not in setup  # that page forces "Acepto"; the AGPL need not be accepted to run it
    assert directives("Messages")["WizardInfoBefore"] == "Licencia"


def test_the_whole_build_folder_is_installed_with_its_license_notices():
    files = entries("Files")
    assert len(files) == 1
    assert files[0]["Source"] == r"{#SourceDir}\*" and files[0]["DestDir"] == "{app}"
    assert {"recursesubdirs", "createallsubdirs", "ignoreversion"} <= set(files[0]["Flags"].split())
    assert {"LEEME.txt", "LICENSE.txt", "LICENSES.md", "THIRD_PARTY_LICENSES/INDEX.txt"} <= set(builder.REQUIRED)


def test_upgrades_replace_the_libraries_and_never_delete_user_data(monkeypatch, tmp_path):
    deleted = {entry["Name"] for entry in entries("InstallDelete")}
    assert r"{app}\_internal" in deleted and r"{app}\THIRD_PARTY_LICENSES" in deleted
    assert all(name.startswith("{app}\\") for name in deleted | {e["Name"] for e in entries("UninstallDelete")})
    # The user's data folder is the app's (server.app_data_dir); only an explicit Yes deletes it.
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert server.app_data_dir() == tmp_path / "Anonimizador"
    code = "\n".join(section("Code"))
    assert r"ExpandConstant('{localappdata}\Anonimizador')" in code
    assert "UninstallSilent" in code and "MB_DEFBUTTON2" in code and "= IDYES then" in code


def test_running_app_blocks_install_and_uninstall():
    assert defines()["AppMutex"] == launcher.APP_MUTEX
    assert directives("Setup")["AppMutex"] == r"{#AppMutex},Global\{#AppMutex}"


def test_shortcuts():
    icons = {entry["Name"]: entry for entry in entries("Icons")}
    assert defines()["StartMenuName"] == "Anonimizador"
    start = icons[r"{autoprograms}\{#StartMenuName}"]
    assert start["Filename"] == r"{app}\{#AppExe}" and "Tasks" not in start
    desktop = icons[r"{autodesktop}\{#StartMenuName}"]
    assert desktop["Tasks"] == "desktopicon"
    task = next(entry for entry in entries("Tasks") if entry["Name"] == "desktopicon")
    assert "unchecked" in task["Flags"].split()


def test_test_builds_are_marked_not_for_distribution():
    assert re.search(r'#ifdef TestBuild\s+#define TestMark "[^"]*no distribuir[^"]*"', SCRIPT)
    setup = directives("Setup")
    for key in ("AppVerName", "UninstallDisplayName", "VersionInfoDescription"):
        assert "{#TestMark}" in setup[key]
    assert "NO LA DISTRIBUYA" in SCRIPT


@pytest.mark.skipif(sys.platform != "win32", reason="Windows mutexes")
def test_launcher_holds_the_app_mutex():
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.OpenMutexW.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR)
    kernel32.OpenMutexW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)

    def exists() -> bool:
        handle = kernel32.OpenMutexW(0x00100000, False, launcher.APP_MUTEX)  # SYNCHRONIZE
        if handle:
            kernel32.CloseHandle(handle)
        return bool(handle)

    before = exists()  # True if the real app is open on this computer
    handles = launcher.hold_app_mutex()
    try:
        assert len(handles) >= 1 and exists()
    finally:
        for handle in handles:
            kernel32.CloseHandle(handle)
    if not before:
        assert not exists()


# --- build_installer.py ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("version", "numeric"),
    [("0.1.0", "0.1.0.0"), ("2", "2.0.0.0"), ("1.2.3.4.5", "1.2.3.4"), ("2.0rc1", "2.0.0.0"), ("dev", "0.0.0.0")],
)
def test_numeric_version(version, numeric):
    assert builder.numeric_version(version) == numeric


def test_iscc_command_passes_the_build_values():
    app_dir = Path(r"C:\build\dist\LightAnonymizer")
    command = builder.iscc_command(Path("ISCC.exe"), "0.1.0", app_dir, app_dir.parent, test_build=False)
    assert command[0] == "ISCC.exe" and command[-1] == str(builder.ISS)
    assert "/DAppVersion=0.1.0" in command and "/DAppNumericVersion=0.1.0.0" in command
    assert f"/DSourceDir={app_dir}" in command
    assert f"/O{app_dir.parent}" in command and "/FLightAnonymizer-0.1.0-setup" in command
    assert not any(arg.startswith("/DTestBuild") for arg in command)
    test = builder.iscc_command(Path("ISCC.exe"), "0.1.0", app_dir, app_dir.parent, test_build=True)
    assert "/DTestBuild=1" in test


def test_find_iscc(monkeypatch, tmp_path):
    for variable in ("ISCC", "LOCALAPPDATA", "ProgramFiles(x86)", "ProgramFiles"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(builder.shutil, "which", lambda name: None)
    assert builder.find_iscc() is None
    assert builder.find_iscc(str(tmp_path / "missing.exe")) is None
    per_user = tmp_path / "Programs" / "Inno Setup 6" / "ISCC.exe"
    per_user.parent.mkdir(parents=True)
    per_user.write_bytes(b"")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert builder.find_iscc() == per_user
    chosen = tmp_path / "other" / "ISCC.exe"
    chosen.parent.mkdir()
    chosen.write_bytes(b"")
    monkeypatch.setenv("ISCC", str(chosen))
    assert builder.find_iscc() == chosen
    assert builder.find_iscc(str(per_user)) == per_user


def fake_build(tmp_path: Path, *, commit="a" * 40, dirty=False, version="0.1.0") -> tuple[Path, Path]:
    """A one-folder build as build_exe.py leaves it (tiny files) and its build record."""
    app = tmp_path / "dist" / "LightAnonymizer"
    (app / "_internal").mkdir(parents=True)
    (app / "THIRD_PARTY_LICENSES").mkdir()
    (app / "LightAnonymizer.exe").write_bytes(b"MZ fake executable")
    (app / "_internal" / "core.dll").write_bytes(b"core")
    for name in ("LEEME.txt", "LICENSE.txt", "LICENSES.md", "THIRD_PARTY_LICENSES/INDEX.txt"):
        (app / name).write_text(name, encoding="utf-8")
    info = app.parent / "LightAnonymizer-build.json"
    builder.write_build_info(info, version, commit, dirty, app / "LightAnonymizer.exe")
    return app, info


def test_build_record_must_describe_the_folder(tmp_path):
    app, info = fake_build(tmp_path)
    record, problems = builder.read_build_info(info, app)
    assert problems == [] and record["version"] == "0.1.0" and record["dirty"] is False
    (app / "LightAnonymizer.exe").write_bytes(b"MZ another build")
    assert any("is not the one" in p for p in builder.read_build_info(info, app)[1])
    (app / "LEEME.txt").unlink()
    assert any(p.endswith("LEEME.txt is missing") for p in builder.read_build_info(info, app)[1])
    info.write_text(json.dumps({"version": "0.1.0"}), encoding="utf-8")
    assert "not a valid build record" in builder.read_build_info(info, app)[1][0]
    info.unlink()
    assert "build_exe.py" in builder.read_build_info(info, app)[1][0]


def test_which_installers_are_test_builds():
    clean = {"commit": "c1", "dirty": False}
    assert builder.is_test_build(clean, changed=False, allow_dirty=False) == (False, None)
    # The build itself is a test build: so is its installer, without asking again.
    assert builder.is_test_build({"commit": "c1", "dirty": True}, changed=True, allow_dirty=False) == (True, None)
    assert builder.is_test_build({"commit": None, "dirty": True}, changed=True, allow_dirty=False) == (True, None)
    # The installer's sources must be those of the build's commit.
    test, refusal = builder.is_test_build(clean, changed=True, allow_dirty=False)
    assert test and "c1" in refusal and "--allow-dirty" in refusal
    assert builder.is_test_build(clean, changed=True, allow_dirty=True) == (True, None)


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_only_the_installer_sources_must_match_the_build_commit(tmp_path):
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True).stdout

    git("init", "-q")
    for name in (*builder.INSTALLER_SOURCES, "README.md"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(name, encoding="utf-8")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-q", "-m", "build")
    commit = git("rev-parse", "HEAD").strip()
    assert not builder.sources_changed(commit, tmp_path)
    (tmp_path / "README.md").write_text("documentation changed after the build", encoding="utf-8")
    assert not builder.sources_changed(commit, tmp_path)
    (tmp_path / "packaging" / "installer.iss").write_text("changed", encoding="utf-8")
    assert builder.sources_changed(commit, tmp_path)
    assert builder.sources_changed("0" * 40, tmp_path)  # unknown commit


def test_build_refuses_without_compiling(monkeypatch, tmp_path, capsys):
    app, info = fake_build(tmp_path)
    monkeypatch.setattr(builder, "sources_changed", lambda commit: True)
    monkeypatch.setattr(builder.subprocess, "run", lambda *a, **k: pytest.fail("ISCC must not run"))
    assert builder.build(Path("ISCC.exe"), app, info) == 1
    assert "--allow-dirty" in capsys.readouterr().err


def test_main_says_how_to_get_inno_setup(monkeypatch, capsys):
    monkeypatch.setattr(builder, "find_iscc", lambda explicit=None: None)
    monkeypatch.setattr(builder.sys, "platform", "win32")
    assert builder.main([]) == 1
    assert "winget install --id JRSoftware.InnoSetup" in capsys.readouterr().err


# --- A real installer of a fake application (needs Inno Setup 6) --------------------------------------

ISCC = builder.find_iscc() if sys.platform == "win32" else None


class Mutex:
    """Holds a named mutex, like a running app."""

    def __init__(self, name: str):
        import ctypes
        from ctypes import wintypes

        self.kernel32 = ctypes.WinDLL("kernel32")
        self.kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
        self.kernel32.CreateMutexW.restype = wintypes.HANDLE
        self.kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        self.name = name

    def __enter__(self):
        self.handle = self.kernel32.CreateMutexW(None, False, self.name)
        assert self.handle
        return self

    def __exit__(self, *exc):
        self.kernel32.CloseHandle(self.handle)


UNINSTALL_KEYS = r"Software\Microsoft\Windows\CurrentVersion\Uninstall"


def uninstall_key(app_id: str):
    import winreg

    try:
        return winreg.OpenKey(winreg.HKEY_CURRENT_USER, rf"{UNINSTALL_KEYS}\{app_id}_is1")
    except FileNotFoundError:
        return None


@pytest.mark.skipif(ISCC is None, reason="needs Inno Setup 6 (ISCC.exe) on Windows")
def test_fake_app_installs_upgrades_and_uninstalls(tmp_path):
    import winreg

    tag = uuid.uuid4().hex[:8]
    test_defines = {
        "AppId": f"LightAnonymizerTest-{tag}",
        "AppMutex": f"LightAnonymizer.Test-{tag}",
        "StartMenuName": f"Anonimizador prueba {tag}",
    }

    def compile_installer(app: Path, version: str, test_build: bool) -> Path:
        out = tmp_path / "out"
        command = builder.iscc_command(ISCC, version, app, out, test_build=test_build, extra=test_defines)
        subprocess.run(command, check=True, capture_output=True)
        return out / f"{builder.installer_name(version)}.exe"

    old_app, _ = fake_build(tmp_path / "v1", version="0.0.1")
    (old_app / "_internal" / "dropped_in_v2.dll").write_bytes(b"old library")
    new_app, _ = fake_build(tmp_path / "v2", version="0.0.2")
    (new_app / "_internal" / "added_in_v2.dll").write_bytes(b"new library")
    old_setup = compile_installer(old_app, "0.0.1", test_build=True)
    new_setup = compile_installer(new_app, "0.0.2", test_build=False)
    assert old_setup.is_file() and new_setup.is_file()

    target = tmp_path / "Programs" / "LightAnonymizer"
    start_menu = Path(os.environ["APPDATA"], "Microsoft", "Windows", "Start Menu", "Programs")

    def install(setup: Path) -> int:
        args = [str(setup), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/NOICONS", f"/DIR={target}"]
        return subprocess.run(args, timeout=120).returncode

    def installed(value: str = "DisplayVersion") -> str | None:
        """What Settings > Apps shows about the test installation (None: not installed)."""
        key = uninstall_key(test_defines["AppId"])
        if key is None:
            return None
        with key:
            return winreg.QueryValueEx(key, value)[0]

    def uninstall() -> int:
        args = [str(target / "unins000.exe"), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"]
        return subprocess.run(args, timeout=120).returncode

    try:
        assert install(old_setup) == 0
        assert (target / "_internal" / "dropped_in_v2.dll").is_file() and (target / "LEEME.txt").is_file()
        assert installed() == "0.0.1"
        # A test build says so in the installed-apps list (and the accents survive the compiler).
        assert installed("DisplayName") == "Anonimizador (compilación de prueba: no distribuir)"
        assert installed("Publisher") == "Gonzalo Fuentes"
        assert not list(start_menu.glob(f"{test_defines['StartMenuName']}*"))  # /NOICONS

        with Mutex(test_defines["AppMutex"]):  # the app is running: Setup gives up, nothing changes
            assert install(new_setup) != 0
        assert (target / "_internal" / "dropped_in_v2.dll").is_file() and installed() == "0.0.1"

        assert install(new_setup) == 0
        assert not (target / "_internal" / "dropped_in_v2.dll").exists()  # no old library left behind
        assert (target / "_internal" / "added_in_v2.dll").is_file()
        assert installed() == "0.0.2" and installed("DisplayName") == "Anonimizador"

        with Mutex(test_defines["AppMutex"]):  # running: the uninstaller gives up too
            assert uninstall() != 0
        assert (target / "LightAnonymizer.exe").is_file() and installed() == "0.0.2"

        (target / "_internal" / "written_later.txt").write_text("x", encoding="utf-8")
        assert uninstall() == 0  # it ends in a copy of itself, which deletes the folder: wait for it
        deadline = time.monotonic() + 60
        while target.exists() and time.monotonic() < deadline:
            time.sleep(0.25)
        assert not target.exists()
        assert installed() is None
    finally:
        if (target / "unins000.exe").exists():
            uninstall()
        key = uninstall_key(test_defines["AppId"])
        if key is not None:
            key.Close()
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, rf"{UNINSTALL_KEYS}\{test_defines['AppId']}_is1")
