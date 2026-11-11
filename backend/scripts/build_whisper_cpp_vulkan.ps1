<#
.SYNOPSIS
    Builds whisper.cpp's `whisper-server` with the Vulkan backend enabled
    (GGML_VULKAN=ON) and installs the binary + its DLLs into
    backend/vendor/whisper-cpp-vulkan/ for local Local-mode dev/testing on
    Windows AMD/Intel GPUs.

.DESCRIPTION
    Mirrors the exact build recipe .github/workflows/release.yml's
    windows-latest leg uses for the shipped binary -- same pinned
    whisper.cpp tag/commit, same -DGGML_VULKAN=ON flag -- so local
    verification and CI never drift onto different whisper.cpp versions.
    See docs/adr/011-whisper-cpp-vulkan-stt-provider.md for the full
    design rationale.

    The CPU code is built for every x86-64 processor, not for the build
    machine's: GGML_NATIVE=OFF plus GGML_BACKEND_DL=ON and
    GGML_CPU_ALL_VARIANTS=ON compile one ggml-cpu-<variant>.dll per CPU
    level (x64, sse42, sandybridge, haswell, skylakex, icelake, alderlake),
    and whisper-server's ggml_backend_load_all() asks each variant's
    ggml_backend_score() -- compiled without the variant's arch flags --
    whether this CPU has its instructions, then loads the best one that
    does. The default GGML_NATIVE=ON compiled ggml-cpu.dll for the GitHub
    runner's CPU, so v1.0.1 shipped AVX-512 code that died with 0xC000001D
    (illegal instruction) on CPUs without AVX-512, e.g. a Ryzen 5 5600.
    This is the same recipe llama.cpp's own Windows release builds use.
    Assert-CpuDispatch below fails the build when that guarantee breaks.

    Prerequisites (not installed by this script):
      - git
      - A Visual Studio 2022 C++ toolchain (Build Tools or full IDE) with
        the "Desktop development with C++" workload -- provides cl.exe,
        and CMake + Ninja are bundled with it even when neither is on PATH.
      - The Vulkan SDK (https://vulkan.lunarg.com/sdk/home#windows),
        providing vulkan-1.lib, headers, and glslc for shader compilation.

    The source checkout and CMake build tree are placed under a SHORT path
    in %TEMP% rather than under this repo -- whisper.cpp's Vulkan backend
    builds a helper tool (vulkan-shaders-gen) via a nested CMake
    ExternalProject, whose own intermediate build paths (e.g.
    ggml\src\ggml-vulkan\vulkan-shaders-gen-prefix\src\...\CMakeFiles\
    CMakeScratch\TryCompile-xxxxx) can exceed Windows' 260-character
    MAX_PATH when nested under this project's already-long path
    ("...\Pet projects\justSay\backend\vendor\..."), which fails the
    build with a cryptic "Failed to set working directory" CMake error --
    confirmed by hitting this exact failure while building on this
    project's own dev box. Only the final binary + DLLs are copied into
    the repo; the source/build tree is disposable.

.PARAMETER WhisperCppTag
    Pinned whisper.cpp git tag. backend/tests/test_build_definitions.py
    reads this parameter's default literal below and compares it against
    backend/scripts/build_whisper_cpp_metal.sh's WHISPER_CPP_TAG -- the two
    platform build recipes must never drift onto different whisper.cpp
    versions, and that test is what enforces it. Also mirrored by
    .github/workflows/release.yml, which invokes this same script.

.PARAMETER Clean
    Delete any existing build tree under %TEMP% first, forcing a full
    reconfigure/rebuild instead of an incremental one.
#>

param(
    [string]$WhisperCppTag = "v1.7.6",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

# Keep this in sync with .github/workflows/release.yml's own pin -- see the
# script docstring above for why both build recipes must never drift apart.
$RepoUrl = "https://github.com/ggerganov/whisper.cpp.git"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$VendorDir = Join-Path $RepoRoot "backend\vendor\whisper-cpp-vulkan"

# Short, flat path -- see the MAX_PATH note in the script docstring.
$WorkDir = Join-Path $env:TEMP "justsay-whisper-cpp-vulkan-build"
$SrcDir = Join-Path $WorkDir "src"
$BuildDir = Join-Path $WorkDir "build"

function Write-Step($msg) {
    Write-Host "==> $msg" -ForegroundColor Cyan
}

function Find-VulkanSdk {
    if ($env:VULKAN_SDK -and (Test-Path (Join-Path $env:VULKAN_SDK "Include\vulkan\vulkan.h"))) {
        return $env:VULKAN_SDK
    }
    # A machine-level VULKAN_SDK env var set by a *just-completed* installer
    # (e.g. via winget) is not visible to an already-running shell/session --
    # it only takes effect for processes started after a fresh login/reboot.
    # Fall back to scanning the SDK's own default install root.
    $candidates = Get-ChildItem -Path "C:\VulkanSDK" -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending
    foreach ($c in $candidates) {
        if (Test-Path (Join-Path $c.FullName "Include\vulkan\vulkan.h")) {
            return $c.FullName
        }
    }
    return $null
}

function Find-VsDevEnv {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw "vswhere.exe not found at '$vswhere'. Install Visual Studio 2022 " +
              "(Build Tools or IDE) with the 'Desktop development with C++' workload."
    }
    $vsInstall = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if (-not $vsInstall) {
        throw "No Visual Studio installation with the C++ toolchain (VC.Tools.x86.x64) was found."
    }
    $vcvars = Join-Path $vsInstall "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vcvars)) {
        throw "vcvars64.bat not found under '$vsInstall'."
    }
    $cmake = Join-Path $vsInstall "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    $ninja = Join-Path $vsInstall "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
    # Not every VS Build Tools install bundles CMake/Ninja (an optional
    # component) -- fall back to a PATH-resolved copy if bundled ones are
    # missing, and fail with a clear message if neither is available.
    if (-not (Test-Path $cmake)) {
        $onPath = Get-Command cmake.exe -ErrorAction SilentlyContinue
        if (-not $onPath) {
            throw "CMake not found (bundled with VS or on PATH). Install the 'C++ CMake tools " +
                  "for Windows' component, or install CMake separately."
        }
        $cmake = $onPath.Source
    }
    if (-not (Test-Path $ninja)) {
        $onPath = Get-Command ninja.exe -ErrorAction SilentlyContinue
        if (-not $onPath) {
            throw "Ninja not found (bundled with VS or on PATH). Install the 'C++ CMake tools " +
                  "for Windows' component, or install Ninja separately."
        }
        $ninja = $onPath.Source
    }
    return [PSCustomObject]@{
        VcVars = $vcvars
        Cmake  = $cmake
        Ninja  = $ninja
    }
}

Write-Step "Checking prerequisites"

if (-not (Get-Command git.exe -ErrorAction SilentlyContinue)) {
    throw "git not found on PATH."
}

$vulkanSdk = Find-VulkanSdk
if (-not $vulkanSdk) {
    throw "Vulkan SDK not found. Install it from https://vulkan.lunarg.com/sdk/home#windows " +
          "(or 'winget install KhronosGroup.VulkanSDK'), then re-run this script " +
          "(a fresh shell may be needed for the VULKAN_SDK env var to take effect)."
}
Write-Host "  Vulkan SDK: $vulkanSdk"

$vs = Find-VsDevEnv
Write-Host "  MSVC toolchain: $($vs.VcVars)"
Write-Host "  CMake: $($vs.Cmake)"
Write-Host "  Ninja: $($vs.Ninja)"

if ($Clean -and (Test-Path $WorkDir)) {
    Write-Step "Removing existing build tree: $WorkDir"
    Remove-Item -Recurse -Force $WorkDir
}

New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null

if (-not (Test-Path (Join-Path $SrcDir ".git"))) {
    Write-Step "Cloning whisper.cpp @ $WhisperCppTag"
    # -c core.longpaths=true: whisper.cpp's repo ships some deeply-nested
    # example asset paths (Android/Swift bindings) that exceed MAX_PATH
    # during checkout otherwise, even though none of them are needed here.
    git -c core.longpaths=true clone --branch $WhisperCppTag --depth 1 $RepoUrl $SrcDir
} else {
    Write-Step "Reusing existing checkout at $SrcDir (pass -Clean to force a fresh clone)"
}

# All build commands run inside one cmd.exe invocation so vcvars64.bat's
# environment (cl.exe on PATH, INCLUDE/LIB set, etc.) is visible to the
# cmake/ninja calls that follow it -- each `Bash`/PowerShell-tool-style
# separate process invocation would lose that environment immediately.
#
# The batch is written to a temp .bat with CRLF endings and run via `cmd /c
# <file>`, NOT `cmd /c <multi-line-string>`. cmd.exe only treats CRLF (not a
# bare LF) as a command separator, and this .ps1 can be checked out with LF
# endings (.gitattributes pins `*.sh` only, so no rule forces CRLF on this
# file and a CI runner gets the LF blob).
# With LF, `cmd /c $string` ran ONLY the first line (`call vcvars`) and silently
# skipped the cmake line -- a 0-exit no-op that produced no binary. A .bat file
# with forced CRLF executes every line regardless of this script's own endings.
function Invoke-VcEnvBatch([string]$Script) {
    $crlf = (($Script -replace "`r?`n", "`r`n").Trim()) + "`r`n"
    $bat = Join-Path ([System.IO.Path]::GetTempPath()) ("justsay-vc-" + [guid]::NewGuid().ToString("N") + ".bat")
    [System.IO.File]::WriteAllText($bat, $crlf, [System.Text.Encoding]::ASCII)
    try { cmd /c "`"$bat`"" } finally { Remove-Item $bat -Force -ErrorAction SilentlyContinue }
}

# The CPU-dispatch contract the shipped engine must keep:
#   - no plain ggml-cpu.dll (one CPU level, loaded unconditionally);
#   - the x64 baseline variant exists, so every x86-64 CPU gets a backend;
#   - every variant exports ggml_backend_score, the gate the loader asks
#     before it keeps a variant;
#   - AVX-512 code appears only in the variants whose score demands AVX-512;
#     whisper-server.exe, whisper.dll, ggml*.dll and the other variants are
#     loaded on every CPU and must hold none.
# The scan matches what only AVX-512 (EVEX) code can contain: zmm registers,
# opmask registers and their k* instructions, and the extra xmm16-31/ymm16-31
# registers. ymm alone is not a finding: the MSVC STL's vectorized algorithms
# ship AVX2 paths behind their own CPUID check, even in the x64 variant. The
# scan must find AVX-512 in the AVX-512 variants, or the scan itself is broken.
$BaselineCpuVariant = "ggml-cpu-x64.dll"
$Avx512CpuVariants = @("ggml-cpu-skylakex.dll", "ggml-cpu-icelake.dll")
$Avx512Pattern = '\bzmm\d+\b|\{k[1-7]\}|\b[xy]mm(1[6-9]|2\d|3[01])\b|' +
    '\bk(and|andn|or|xor|xnor|not|test|ortest|shiftl|shiftr|unpck|add|mov)(b|w|d|q|bw|wd|dq)\b'

function Get-Avx512LineCount([string]$Binary) {
    $listing = Join-Path ([System.IO.Path]::GetTempPath()) ("justsay-disasm-" + [guid]::NewGuid().ToString("N") + ".txt")
    try {
        Invoke-VcEnvBatch @"
call "$($vs.VcVars)" >nul 2>&1
dumpbin /nologo /disasm /out:"$listing" "$Binary"
"@ | Out-Null
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $listing)) {
            throw "dumpbin could not disassemble $Binary (exit $LASTEXITCODE)."
        }
        return @(Select-String -Path $listing -Pattern $Avx512Pattern -CaseSensitive).Count
    } finally {
        Remove-Item $listing -Force -ErrorAction SilentlyContinue
    }
}

function Test-ExportsScore([string]$Binary) {
    $lines = Invoke-VcEnvBatch @"
call "$($vs.VcVars)" >nul 2>&1
dumpbin /nologo /exports "$Binary"
"@
    return @($lines | Where-Object { $_ -match '\sggml_backend_score$' }).Count -gt 0
}

function Assert-CpuDispatch([string]$Dir) {
    if (Test-Path (Join-Path $Dir "ggml-cpu.dll")) {
        throw "ggml-cpu.dll was built: the CPU backend is fixed to one CPU level instead of " +
              "being chosen at load time. Build with GGML_BACKEND_DL=ON and GGML_CPU_ALL_VARIANTS=ON."
    }
    if (-not (Test-Path (Join-Path $Dir $BaselineCpuVariant))) {
        throw "$BaselineCpuVariant is missing: a CPU without SSE4.2/AVX would get no CPU backend."
    }
    $variants = Get-ChildItem -Path $Dir -Filter "ggml-cpu-*.dll"
    foreach ($variant in $variants) {
        if (-not (Test-ExportsScore $variant.FullName)) {
            throw "$($variant.Name) exports no ggml_backend_score, so the loader cannot tell " +
                  "whether this CPU can run it."
        }
    }
    foreach ($name in $Avx512CpuVariants) {
        $path = Join-Path $Dir $name
        if (-not (Test-Path $path)) {
            throw "$name is missing: the AVX-512 variant list no longer matches what ggml builds."
        }
        $count = Get-Avx512LineCount $path
        Write-Host "  $name (kept only on AVX-512 CPUs): $count AVX-512 instructions"
        if ($count -eq 0) {
            throw "The AVX-512 scan found nothing in $name, which is built for AVX-512, " +
                  "so the scan is broken and cannot vouch for the other files."
        }
    }
    $loadedEverywhere = Get-ChildItem -Path $Dir -File |
        Where-Object { $_.Extension -in ".exe", ".dll" -and $Avx512CpuVariants -notcontains $_.Name }
    foreach ($binary in $loadedEverywhere) {
        $count = Get-Avx512LineCount $binary.FullName
        Write-Host "  $($binary.Name): $count AVX-512 instructions"
        if ($count -ne 0) {
            throw "$($binary.Name) contains $count AVX-512 instructions but can be loaded on a " +
                  "CPU without AVX-512, where it would crash with 0xC000001D (illegal instruction)."
        }
    }
}

Write-Step "Configuring (CMake + Ninja, GGML_VULKAN=ON)"

$configureCmd = @"
call "$($vs.VcVars)" >nul
set VULKAN_SDK=$vulkanSdk
set PATH=%VULKAN_SDK%\Bin;%PATH%
"$($vs.Cmake)" -S "$SrcDir" -B "$BuildDir" -G Ninja -DCMAKE_MAKE_PROGRAM="$($vs.Ninja)" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_VULKAN=ON -DWHISPER_SDL2=OFF -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=ON -DWHISPER_BUILD_SERVER=ON
"@
Invoke-VcEnvBatch $configureCmd
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed (exit $LASTEXITCODE)." }

Write-Step "Building whisper-server (this can take several minutes)"

# Every DLL is relinked into an emptied bin folder, so one an older recipe
# left behind in a reused build tree (a plain ggml-cpu.dll) cannot be shipped.
Get-ChildItem -Path (Join-Path $BuildDir "bin") -Filter "*.dll" -ErrorAction SilentlyContinue |
    Remove-Item -Force

$buildCmd = @"
call "$($vs.VcVars)" >nul
set VULKAN_SDK=$vulkanSdk
set PATH=%VULKAN_SDK%\Bin;%PATH%
"$($vs.Cmake)" --build "$BuildDir" --target whisper-server --config Release -j $env:NUMBER_OF_PROCESSORS
"@
Invoke-VcEnvBatch $buildCmd
if ($LASTEXITCODE -ne 0) { throw "Build failed (exit $LASTEXITCODE)." }

$binDir = Join-Path $BuildDir "bin"
$serverExe = Join-Path $binDir "whisper-server.exe"
if (-not (Test-Path $serverExe)) {
    throw "Build reported success but whisper-server.exe was not found at '$serverExe'."
}

Write-Step "Installing binary + DLLs into $VendorDir"
# Emptied first: a DLL left over from an older recipe (a plain ggml-cpu.dll
# built for one CPU) would otherwise ship beside the new variants.
if (Test-Path $VendorDir) { Remove-Item -Recurse -Force $VendorDir }
New-Item -ItemType Directory -Force -Path $VendorDir | Out-Null
Copy-Item -Path $serverExe -Destination $VendorDir -Force
Get-ChildItem -Path $binDir -Filter "*.dll" | ForEach-Object {
    Copy-Item -Path $_.FullName -Destination $VendorDir -Force
    Write-Host "  Copied $($_.Name)"
}

Write-Step "Verifying the CPU code runs on every x86-64 processor"
Assert-CpuDispatch $VendorDir

Write-Step "Verifying the build is Vulkan-capable"
Write-Host "(startup log below should name a Vulkan device -- e.g. an AMD/Intel/NVIDIA GPU)" -ForegroundColor Yellow
# whisper-server.exe logs its Vulkan device-enumeration line to stderr on
# startup. Deliberately NOT merging stderr into the success stream (no
# `2>&1`) -- with $ErrorActionPreference = "Stop" (set globally above),
# PowerShell 7 promotes a merged native-command stderr line into a
# terminating NativeCommandError even though the process itself exits
# fine. Left unmerged, stderr just prints straight through to the console
# as intended, without going through PowerShell's error machinery at all.
& (Join-Path $VendorDir "whisper-server.exe") --help | Out-Null

Write-Host ""
Write-Host "Done. whisper-server.exe + DLLs installed at:" -ForegroundColor Green
Write-Host "  $VendorDir"
Write-Host "The GGML model itself is NOT bundled here -- WhisperCppVulkanSTTProvider"
Write-Host "downloads it lazily on first use, same as the other Local-mode providers."
