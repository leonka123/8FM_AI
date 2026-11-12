# Build a standalone Windows executable for the 8FM desktop assistant.
# Usage:
#   python -m pip install pyinstaller
#   powershell -ExecutionPolicy Bypass -File build_app.ps1
#
# The result is dist\8FM-Assistant.exe. It bundles models/lu_luna.json, so the
# executable can be copied to another Windows machine without Python installed.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$python = (Get-Command python).Source
Write-Host "Python: $python"

if (-not (Test-Path "models\lu_luna.json")) {
    throw "models\lu_luna.json not found; build must run from the project root"
}

# Generate the application icon from the same drawing used by the tray.
New-Item -ItemType Directory -Force -Path "build" | Out-Null

# Windows file properties (right click -> Properties -> Details).
@"
VSVersionInfo(
  ffi=FixedFileInfo(filevers=(1,0,0,0), prodvers=(1,0,0,0), mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0, date=(0,0)),
  kids=[
    StringFileInfo([StringTable('040904B0', [StringStruct('CompanyName','leonka'), StringStruct('FileDescription','8FM AI Assistant'), StringStruct('FileVersion','1.0.0'), StringStruct('InternalName','8FM-Assistant'), StringStruct('OriginalFilename','8FM-Assistant.exe'), StringStruct('ProductName','8FM AI'), StringStruct('ProductVersion','1.0.0')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"@ | Set-Content -Path "build\version_info.txt" -Encoding UTF8

& $python -X utf8 -c @"
from PIL import Image
from eightfm.desktop_app import _tray_image

image = _tray_image().resize((256, 256), Image.LANCZOS)
image.save('build/app.ico', sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print('icon: build/app.ico')
"@

& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --windowed `
    --name "8FM-Assistant" `
    --icon "build\app.ico" `
    --version-file "build\version_info.txt" `
    --add-data "models\lu_luna.json;models" `
    --hidden-import "pystray._win32" `
    --collect-submodules "pystray" `
    "run_assistant_app.py"

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Built: dist\8FM-Assistant.exe"
Write-Host "Run it, fill in api_id/api_hash/phone, then log in with the Telegram code."
