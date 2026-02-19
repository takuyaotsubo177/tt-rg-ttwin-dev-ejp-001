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
    @("..\..\..\common\app\twinner_bot\.vscode", ".\.vscode"),
    @("..\..\..\common\app\twinner_bot\core", ".\core"),
    @("..\..\..\common\app\twinner_bot\template", ".\template"),
    @("..\..\..\common\app\twinner_bot\app.py", ".\app.py"),
    @("..\..\..\common\app\twinner_bot\Dockerfile", ".\Dockerfile"),
    @("..\..\..\common\app\twinner_bot\requirements.txt", ".\requirements.txt")
)
Copy-PathPairs $PathPairs