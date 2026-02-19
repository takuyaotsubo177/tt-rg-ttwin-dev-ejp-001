$CWD = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent
Set-Location -Path $CWD

# -------------------------------------------------- util
function Copy-PathPairs {
    Param (
        [Parameter(Mandatory=$true)]
        [array]$PathPairs
    )

    foreach ($Pair in $PathPairs) {
        $Src = $Pair[0]
        $Dst = $Pair[1]

        if (Test-Path $Dst) {
            Write-Host "Skipped $Src (already exists at $Dst)"
            continue
        }

        $DstDir = Split-Path $Dst
        if (-not (Test-Path $DstDir)) {
            New-Item -ItemType Directory -Path $DstDir | Out-Null
        }

        if (Test-Path $Src -PathType Leaf) {
            Copy-Item -Path $Src -Destination $Dst -Force
        } else {
            Copy-Item -Path $Src -Destination $Dst -Recurse -Force
        }
        Write-Host "Copied $Src to $Dst"
    }
}

# -------------------------------------------------- main
# cp required files
$PathPairs = @(
    @("..\..\common\data_pipeline\.vscode", ".\.vscode"),
    @("..\..\common\data_pipeline\src\core", ".\src\core"),
    @("..\..\common\data_pipeline\src\function_app.py", ".\src\function_app.py"),
    @("..\..\common\data_pipeline\src\host.json", ".\src\host.json"),
    @("..\..\common\data_pipeline\src\requirements.txt", ".\src\requirements.txt"),
    @("..\..\common\data_pipeline\.funcignore", ".\.funcignore")
)
Copy-PathPairs $PathPairs