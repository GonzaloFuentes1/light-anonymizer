; Inno Setup 6 script of the Windows installer: installs the one-folder build (dist\LightAnonymizer)
; for the current user, without administrator rights.
;
;     uv run python scripts/build_installer.py      (after build_exe.py, from the same commit)
;
; build_installer.py passes these values with /D on ISCC's command line; compiling this file by hand
; without them stops with an error:
;
;   AppVersion         version of the build (pyproject.toml), e.g. 0.1.0
;   AppNumericVersion  the same with four numbers, for the file properties, e.g. 0.1.0.0
;   SourceDir          the one-folder build: LightAnonymizer.exe, _internal\, LEEME.txt, LICENSE.txt,
;                      LICENSES.md and THIRD_PARTY_LICENSES\ (the AGPL notices travel with the program)
;   TestBuild          defined for a build that is not exactly a commit: marked "no distribuir", and
;                      its file name ends in -PRUEBA-no-distribuir
;   AppId, AppMutex, StartMenuName
;                      only overridden by tests/test_installer.py, so that a test installation never
;                      touches a real one
;
; Decisions (README, "Building the Windows executable"):
; - Per user, in %LOCALAPPDATA%\Programs\LightAnonymizer: no administrator, and a short path outside
;   OneDrive (native libraries can fail to load from long paths).
; - Upgrades install over the previous version (same AppId and folder). _internal\ and
;   THIRD_PARTY_LICENSES\ are deleted first, so old and new libraries never mix.
; - The user's data (%LOCALAPPDATA%\Anonimizador: technical log, time estimates, WebView2 profiles)
;   is outside the program folder: installing never touches it, and uninstalling keeps it unless the
;   user answers Yes to a question (No is the default; silent uninstalls always keep it).
; - Uninstalling always deletes the working copies of documents an app that did not close normally
;   left in %TEMP% (anonimizador_session_* of processes that are gone, and anonimizador_export_*
;   when no session is alive), as the app itself does when it starts.
; - The license is shown on a page that needs no "I accept": the AGPL (section 9) does not have to be
;   accepted to receive or run the program, only to modify or convey it.
; - While the app runs it holds the mutex AppMutex (packaging/launcher.py): Setup and Uninstall ask
;   to close it first.
; - User-facing texts are Spanish; they use "usted", like Inno Setup's Spanish messages around them.

#ifndef AppVersion
  #error Compile with scripts/build_installer.py, which defines AppVersion, AppNumericVersion and SourceDir
#endif
#ifndef AppNumericVersion
  #error AppNumericVersion is not defined (use scripts/build_installer.py)
#endif
#ifndef SourceDir
  #error SourceDir is not defined (use scripts/build_installer.py)
#endif
#ifndef AppId
  ; Never change it: Windows recognizes the installed program, and so the upgrades, by this value.
  #define AppId "{{685C4DB7-5B07-43D5-8BA0-4BAB4846D1AD}"
#endif
#ifndef AppMutex
  #define AppMutex "LightAnonymizer.Running"
#endif
#ifndef StartMenuName
  #define StartMenuName "Anonimizador"
#endif
#define AppExe "LightAnonymizer.exe"
#define Repository "https://github.com/GonzaloFuentes1/light-anonymizer"
#ifdef TestBuild
  #define TestMark " (compilación de prueba: no distribuir)"
  ; Same as TEST_SUFFIX in scripts/build_installer.py.
  #define OutputSuffix "-PRUEBA-no-distribuir"
#else
  #define TestMark ""
  #define OutputSuffix ""
#endif

[Setup]
AppId={#AppId}
AppName=Anonimizador
AppVersion={#AppVersion}
AppVerName=Anonimizador {#AppVersion}{#TestMark}
AppPublisher=Gonzalo Fuentes
AppPublisherURL={#Repository}
AppSupportURL={#Repository}
AppUpdatesURL={#Repository}/releases
AppCopyright=© 2026 Gonzalo Fuentes. Licencia GNU AGPL-3.0-or-later.
AppMutex={#AppMutex},Global\{#AppMutex}
VersionInfoVersion={#AppNumericVersion}
VersionInfoProductName=Light Anonymizer
VersionInfoProductTextVersion={#AppVersion}{#TestMark}
VersionInfoDescription=Instalador del Anonimizador{#TestMark}
VersionInfoCompany=Gonzalo Fuentes
VersionInfoCopyright=© 2026 Gonzalo Fuentes. Licencia GNU AGPL-3.0-or-later.

PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\LightAnonymizer
DisableDirPage=yes
; No folder page, but the last page before installing says where the program goes.
AlwaysShowDirOnReadyPage=yes
DisableProgramGroupPage=yes
; Only so that /NOICONS (tests, scripted installs) can skip the Start menu shortcut.
AllowNoIcons=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
InfoBeforeFile=..\LICENSE
ShowLanguageDialog=no
WizardStyle=modern
#ifdef TestBuild
DisableWelcomePage=no
#endif

CloseApplications=yes
RestartApplications=no
UninstallDisplayName=Anonimizador{#TestMark}
UninstallDisplayIcon={app}\{#AppExe}

OutputDir=..\dist
OutputBaseFilename=LightAnonymizer-{#AppVersion}-setup{#OutputSuffix}
Compression=lzma2/max
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=2

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"

[Messages]
WizardInfoBefore=Licencia
InfoBeforeLabel=El Anonimizador es software libre (GNU AGPL v3 o posterior) y se entrega sin ninguna garantía.
InfoBeforeClickLabel=No necesita aceptar esta licencia para instalar ni usar el programa: rige al copiarlo, modificarlo o distribuirlo (el código fuente se indica en LEEME.txt). Haga clic en Siguiente para continuar.
#ifdef TestBuild
WelcomeLabel2=Este programa instalará [name/ver] en su sistema.%n%nATENCIÓN: es una compilación de prueba, hecha con cambios que no están en el código fuente publicado. NO LA DISTRIBUYA.
#endif

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[InstallDelete]
; Upgrades: the previous version's libraries and license texts go before the new ones are copied.
Type: filesandordirs; Name: "{app}\_internal"
Type: filesandordirs; Name: "{app}\THIRD_PARTY_LICENSES"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#StartMenuName}"; Filename: "{app}\{#AppExe}"; WorkingDir: "{app}"; Comment: "Anonimiza documentos PDF e imágenes en este computador"; Check: not WizardNoIcons
Name: "{autodesktop}\{#StartMenuName}"; Filename: "{app}\{#AppExe}"; WorkingDir: "{app}"; Comment: "Anonimiza documentos PDF e imágenes en este computador"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Abrir el Anonimizador"; Flags: nowait postinstall skipifsilent
Filename: "{app}\LEEME.txt"; Description: "Leer LEEME.txt (uso, licencia y código fuente)"; Flags: nowait postinstall skipifsilent shellexec unchecked

[UninstallDelete]
; Anything the program files gained after installing (the user's data is elsewhere).
Type: filesandordirs; Name: "{app}\_internal"

[Code]
const
  PROCESS_QUERY_LIMITED_INFORMATION = $1000;
  STILL_ACTIVE = 259;
  ERROR_ACCESS_DENIED = 5;

function OpenProcess(Access: Longword; Inherit: Bool; Pid: Longword): Longword;
external 'OpenProcess@kernel32.dll stdcall';
function GetExitCodeProcess(Process: Longword; var ExitCode: Longword): Bool;
external 'GetExitCodeProcess@kernel32.dll stdcall';
function CloseHandle(Handle: Longword): Bool;
external 'CloseHandle@kernel32.dll stdcall';

// The user's data folder (anonymizer/api/server.py, app_data_dir): kept unless the user asks.
function UserDataDir(): String;
begin
  Result := ExpandConstant('{localappdata}\Anonimizador');
end;

// Same rule as pid_alive in anonymizer/api/server.py.
function ProcessAlive(Pid: Longword): Boolean;
var
  Process, ExitCode: Longword;
begin
  Process := OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, Pid);
  if Process = 0 then begin
    Result := DLLGetLastError() = ERROR_ACCESS_DENIED;  // it exists, but belongs to someone else
    Exit;
  end;
  Result := False;
  if GetExitCodeProcess(Process, ExitCode) then
    Result := ExitCode = STILL_ACTIVE;
  CloseHandle(Process);
end;

// A session folder whose process still runs (a copy of the app this installer does not know of:
// the zip, a development run). Without a readable pid file it is stale, as in cleanup_stale_sessions.
function SessionAlive(Folder: String): Boolean;
var
  Text: AnsiString;
  Pid: Integer;
begin
  Result := False;
  if LoadStringFromFile(Folder + '\pid', Text) then begin
    Pid := StrToIntDef(Trim(String(Text)), 0);
    Result := (Pid > 0) and ProcessAlive(Pid);
  end;
end;

// Working copies of documents left in %TEMP% by an app that was killed or crashed. The app deletes
// them at its next start, which may never come after uninstalling. Export folders record no owner:
// they go only when no session at all is alive.
procedure DeleteStaleWorkingCopies();
var
  Temp: String;
  Found: TFindRec;
  Stale: TStringList;
  AnyAlive: Boolean;
  I: Integer;
begin
  Temp := AddBackslash(GetTempDir());
  Stale := TStringList.Create;
  AnyAlive := False;
  try
    if FindFirst(Temp + 'anonimizador_session_*', Found) then begin
      try
        repeat
          if (Found.Attributes and FILE_ATTRIBUTE_DIRECTORY) <> 0 then begin
            if SessionAlive(Temp + Found.Name) then
              AnyAlive := True
            else
              Stale.Add(Temp + Found.Name);
          end;
        until not FindNext(Found);
      finally
        FindClose(Found);
      end;
    end;
    for I := 0 to Stale.Count - 1 do
      DelTree(Stale[I], True, True, True);
    if not AnyAlive then
      DelTree(Temp + 'anonimizador_export_*', False, True, True);
  finally
    Stale.Free;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Folder, Temp: String;
begin
  if CurUninstallStep <> usPostUninstall then
    Exit;
  DeleteStaleWorkingCopies();  // also when silent: they are temporary, and may hold documents
  Folder := UserDataDir();
  if UninstallSilent or not DirExists(Folder) then
    Exit;
  if MsgBox('El Anonimizador guardó en este computador su registro técnico y sus estimaciones de ' +
            'tiempo (nunca documentos), en:' + #13#10#13#10 + Folder + #13#10#13#10 +
            '¿Desea borrarlos también? Si elige No, se conservan para una próxima instalación.',
            mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then begin
    DelTree(Folder, True, True, True);
    // Where the app writes them when %LOCALAPPDATA% cannot be written (server.py, app.py).
    Temp := AddBackslash(GetTempDir());
    DelTree(Temp + 'anonimizador_logs', True, True, True);
    DelTree(Temp + 'anonimizador_webview', True, True, True);
  end;
end;
