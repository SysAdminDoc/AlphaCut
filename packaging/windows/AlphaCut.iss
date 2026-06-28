#define MyAppName "AlphaCut"
#define MyAppPublisher "SysAdminDoc"
#define MyAppURL "https://github.com/SysAdminDoc/AlphaCut"

#ifndef AppVersion
#define AppVersion "1.6.1"
#endif

#ifndef SourceExe
#define SourceExe "..\..\dist\AlphaCut-windows.exe"
#endif

#ifndef OutputDir
#define OutputDir "..\..\dist\installer"
#endif

[Setup]
AppId={{0CB97942-95F9-4FB8-A3CF-38244DDA2651}
AppName={#MyAppName}
AppVersion={#AppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={localappdata}\Programs\AlphaCut
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=..\..\LICENSE
OutputDir={#OutputDir}
OutputBaseFilename=AlphaCut-Setup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
UninstallDisplayIcon={app}\AlphaCut.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "{#SourceExe}"; DestDir: "{app}"; DestName: "AlphaCut.exe"; Flags: ignoreversion
Source: "..\..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\AlphaCut"; Filename: "{app}\AlphaCut.exe"
Name: "{group}\Uninstall AlphaCut"; Filename: "{uninstallexe}"
Name: "{autodesktop}\AlphaCut"; Filename: "{app}\AlphaCut.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\AlphaCut.exe"; Description: "Launch AlphaCut"; Flags: nowait postinstall skipifsilent
