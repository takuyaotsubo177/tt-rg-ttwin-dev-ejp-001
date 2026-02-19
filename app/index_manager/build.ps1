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
    @("..\..\..\common\app\index_manager\.vscode", ".\.vscode"),
    @("..\..\..\common\app\index_manager\core", ".\core"),
    @("..\..\..\common\app\index_manager\app.py", ".\app.py"),
    @("..\..\..\common\app\index_manager\Dockerfile", ".\Dockerfile"),
    @("..\..\..\common\app\index_manager\requirements.txt", ".\requirements.txt")
)
Copy-PathPairs $PathPairs