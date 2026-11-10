<#
  ClipForge — installer + launcher (English, with progress).

  ClipForge.bat downloads and runs this. On the FIRST run it fetches everything it
  needs (private Python, all libraries + GPU acceleration, ffmpeg) and then starts
  the app. The AI model + fonts are fetched by the app on first use. On later runs
  it skips straight to launch. Everything installs into the same folder as
  ClipForge.bat. If you have an NVIDIA card, "Auto" uses the GPU; otherwise CPU.
#>

$ErrorActionPreference = "Stop"
$ProgressPreference = "Continue"     # show download progress bars
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# --- Config ----------------------------------------------------------------
$PyVer  = "3.11.9"
$Repo   = "Ai-Haris/clipforge"
$Branch = "main"
$ZipUrl = "https://github.com/$Repo/archive/refs/heads/$Branch.zip"
$Total  = 6

$Base = $PSScriptRoot
if (-not $Base) { $Base = (Get-Location).Path }
Set-Location $Base
$Py = Join-Path $Base "python\python.exe"

# --- Helpers ---------------------------------------------------------------
function Step($n, $title) {
  Write-Progress -Id 0 -Activity "ClipForge setup" -Status "Step $n of $Total  -  $title" `
                 -PercentComplete ([int]((($n - 1) / $Total) * 100))
  Write-Host ""
  Write-Host ("[{0}/{1}] {2}" -f $n, $Total, $title) -ForegroundColor Cyan
}
function OK($m)   { Write-Host "      $([char]0x2713) $m" -ForegroundColor Green }
function Warn($m) { Write-Host "      ! $m" -ForegroundColor Yellow }

function Get-File($url, $out, $name) {
  Write-Host "      downloading $name ..." -ForegroundColor DarkGray
  Invoke-WebRequest -UseBasicParsing $url -OutFile $out
}

# Run a pip command quietly, showing a live spinner + elapsed time instead of the
# confusing wall of "Collecting..." lines. On failure, print the captured log.
function Invoke-Pip($arguments, $label) {
  $log = Join-Path $env:TEMP ("cf_pip_{0}.log" -f ([guid]::NewGuid().ToString('N').Substring(0,8)))
  $err = "$log.err"
  $p = Start-Process -FilePath $Py -ArgumentList $arguments -NoNewWindow -PassThru `
                     -RedirectStandardOutput $log -RedirectStandardError $err
  $spin = '|', '/', '-', '\'; $i = 0; $t0 = Get-Date
  while (-not $p.HasExited) {
    $el = [int]((Get-Date) - $t0).TotalSeconds
    Write-Host ("`r      {0}  {1} ... {2}s   " -f $spin[$i % 4], $label, $el) -NoNewline -ForegroundColor DarkGray
    Start-Sleep -Milliseconds 250; $i++
  }
  Write-Host ("`r" + (' ' * 70) + "`r") -NoNewline
  if ($p.ExitCode -ne 0) {
    Warn "$label failed. Details:"
    if (Test-Path $err) { Get-Content $err -Tail 25 | ForEach-Object { Write-Host "      $_" -ForegroundColor DarkRed } }
    if (Test-Path $log) { Get-Content $log -Tail 15 | ForEach-Object { Write-Host "      $_" -ForegroundColor DarkGray } }
    throw "$label failed (exit code $($p.ExitCode))."
  }
}

# --- Banner ----------------------------------------------------------------
Write-Host ""
Write-Host "  ================================================" -ForegroundColor Magenta
Write-Host "    ClipForge  -  setup and launch" -ForegroundColor Magenta
Write-Host "  ================================================" -ForegroundColor Magenta
Write-Host "  Install folder: $Base"
Write-Host "  First run downloads everything (needs internet)."
Write-Host "  Please keep this window open."

# --- 1) Private Python -----------------------------------------------------
Step 1 "Setting up a private Python (one-time)"
if (-not (Test-Path $Py)) {
  $z = Join-Path $env:TEMP "cf_py.zip"
  Get-File "https://www.python.org/ftp/python/$PyVer/python-$PyVer-embed-amd64.zip" $z "Python $PyVer"
  Expand-Archive $z (Join-Path $Base "python") -Force
  $pth = Get-ChildItem (Join-Path $Base "python") -Filter "python*._pth" | Select-Object -First 1
  (Get-Content $pth.FullName) -replace '^\s*#\s*import site\s*$', 'import site' | Set-Content $pth.FullName -Encoding Ascii
  if (-not (Select-String -Path $pth.FullName -Pattern '^import site' -Quiet)) {
    Add-Content $pth.FullName "import site" -Encoding Ascii
  }
  $gp = Join-Path $env:TEMP "cf_getpip.py"
  Get-File "https://bootstrap.pypa.io/get-pip.py" $gp "pip"
  Invoke-Pip @($gp, "--no-warn-script-location") "Installing pip"
  OK "Python ready"
} else { OK "Already set up" }

# --- 2) App code -----------------------------------------------------------
Step 2 "Downloading the ClipForge app"
try {
  $z = Join-Path $env:TEMP "cf_app.zip"
  Get-File $ZipUrl $z "app"
  $ex = Join-Path $env:TEMP "cf_app"; if (Test-Path $ex) { Remove-Item $ex -Recurse -Force }
  Expand-Archive $z $ex -Force
  $src = Get-ChildItem $ex -Directory | Select-Object -First 1
  foreach ($d in @("app", "assets", "web", "requirements.txt", "README.md")) {
    $s = Join-Path $src.FullName $d
    if (Test-Path $s) { Copy-Item $s $Base -Recurse -Force }
  }
  OK "App up to date"
} catch {
  if (Test-Path (Join-Path $Base "app\main.py")) { Warn "Couldn't update from GitHub - using the copy already here." }
  else { throw }
}
if (-not (Test-Path (Join-Path $Base "web\dist\index.html"))) { Warn "web\dist missing - the UI may not load." }

# --- 3 & 4) Libraries (one-time; a marker makes re-runs instant) -----------
$marker = Join-Path $Base ".deps_ok"
$needLibs = -not (Test-Path $marker)

Step 3 "Installing core libraries (a few minutes)"
if ($needLibs) {
  Invoke-Pip @("-m", "pip", "install", "--no-warn-script-location", "-r", (Join-Path $Base "requirements.txt")) "Installing core libraries"
  OK "Core libraries installed"
} else { OK "Already installed" }

Step 4 "Installing GPU acceleration (large ~1.3 GB, please be patient)"
if ($needLibs) {
  Invoke-Pip @("-m", "pip", "install", "--no-warn-script-location",
    "faster-whisper==1.2.1", "ctranslate2==4.8.0",
    "nvidia-cublas-cu12==12.9.2.10", "nvidia-cuda-nvrtc-cu12==12.9.86", "nvidia-cudnn-cu12==9.23.2.1") "Installing GPU acceleration"
  "ok" | Out-File $marker -Encoding Ascii
  OK "GPU acceleration installed"
} else { OK "Already installed" }

# --- 5) ffmpeg -------------------------------------------------------------
Step 5 "Setting up the video engine (ffmpeg)"
if (-not (Test-Path (Join-Path $Base "ffmpeg\ffmpeg.exe"))) {
  $z = Join-Path $env:TEMP "cf_ff.zip"
  Get-File "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" $z "ffmpeg"
  $fx = Join-Path $env:TEMP "cf_ff"; if (Test-Path $fx) { Remove-Item $fx -Recurse -Force }
  Expand-Archive $z $fx -Force
  New-Item -ItemType Directory -Force (Join-Path $Base "ffmpeg") | Out-Null
  Copy-Item (Get-ChildItem $fx -Recurse -Filter ffmpeg.exe  | Select-Object -First 1).FullName (Join-Path $Base "ffmpeg\ffmpeg.exe") -Force
  $fp = Get-ChildItem $fx -Recurse -Filter ffprobe.exe | Select-Object -First 1
  if ($fp) { Copy-Item $fp.FullName (Join-Path $Base "ffmpeg\ffprobe.exe") -Force }
  OK "Video engine ready"
} else { OK "Already set up" }

# --- 6) Launch -------------------------------------------------------------
Step 6 "Starting ClipForge"
Write-Progress -Id 0 -Activity "ClipForge setup" -Completed

$env:PATH = (Join-Path $Base "ffmpeg") + ";" + (Join-Path $Base "python") + ";" + $env:PATH
$env:HF_HOME = Join-Path $Base "models"
$env:HF_HUB_DISABLE_TELEMETRY = "1"

# Open the browser only once the server is actually ready (first run downloads the
# ~1.5 GB AI model before it starts serving, so a fixed delay would show an error).
$waiter = @'
$u = "http://127.0.0.1:8000"
for ($i = 0; $i -lt 900; $i++) {
  try { if ((Invoke-WebRequest "$u/health" -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200) { Start-Process $u; break } } catch {}
  Start-Sleep -Seconds 2
}
'@
Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', $waiter

Write-Host ""
Write-Host "  ------------------------------------------------" -ForegroundColor Green
Write-Host "   ClipForge is starting." -ForegroundColor Green
Write-Host "   First run: it downloads the AI model (~1.5 GB)." -ForegroundColor Green
Write-Host "   You'll see download progress below." -ForegroundColor Green
Write-Host "   The browser opens automatically when it's ready." -ForegroundColor Green
Write-Host "   Keep this window open. Close it to stop ClipForge." -ForegroundColor Green
Write-Host "  ------------------------------------------------" -ForegroundColor Green
Write-Host ""

& $Py -m uvicorn app.main:app --host 127.0.0.1 --port 8000
