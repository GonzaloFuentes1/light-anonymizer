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
;   TestBuild          defined for a build that is not exactly a commit: marked "no distribuir"
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
#else
  #define TestMark ""
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
VersionInfoProductTextVersion={#AppVersion}
VersionInfoDescription=Instalador del Anonimizador{#TestMark}
VersionInfoCompany=Gonzalo Fuentes
VersionInfoCopyright=© 2026 Gonzalo Fuentes. Licencia GNU AGPL-3.0-or-later.

PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\LightAnonymizer
DisableDirPage=yes
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
OutputBaseFilename=LightAnonymizer-{#AppVersion}-setup
Compression=lzma2/max
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=2

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"

[Messages]
WizardInfoBefore=Licencia
InfoBeforeLabel=El Anonimizador es software libre (GNU AGPL v3 o posterior) y se entrega sin ninguna garantía.
InfoBeforeClickLabel=No necesita aceptar esta licencia para instalar ni usar el programa: rige al copiarlo, modificarlo o distribuirlo (dónde está el código fuente: LEEME.txt). Haga clic en Siguiente para continuar.
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
// The user's data folder (anonymizer/api/server.py, app_data_dir): kept unless the user asks.
function UserDataDir(): String;
begin
  Result := ExpandConstant('{localappdata}\Anonimizador');
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Folder: String;
begin
  if CurUninstallStep <> usPostUninstall then
    Exit;
  Folder := UserDataDir();
  if UninstallSilent or not DirExists(Folder) then
    Exit;
  if MsgBox('El Anonimizador guardó en este computador su registro técnico y sus estimaciones de ' +
            'tiempo (nunca documentos), en:' + #13#10#13#10 + Folder + #13#10#13#10 +
            '¿Desea borrarlos también? Si elige No, se conservan para una próxima instalación.',
            mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
    DelTree(Folder, True, True, True);
end;
