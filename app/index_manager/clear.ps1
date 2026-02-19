function Compare-PathPairs {
    Param (
        [Parameter(Mandatory=$true)]
        [array]$PathPairs
    )

    foreach ($Pair in $PathPairs) {
        $Org = $Pair[0]
        $Rm = $Pair[1]

        if (-not (Test-Path $Rm)) {
            continue
        } elseif (Compare-Object (Get-Content $Org) (Get-Content $Rm)) {
            Write-Host "Error diff $Rm, so check it out."
            exit 1
        }
    }
}

function Remove-Paths {
    Param (
        [Parameter(Mandatory=$true)]
        [array]$Paths
    )

    foreach ($Rm in $Paths) {
        if (-not (Test-Path $Rm)) {
            continue
        }

        if (Test-Path $Rm -PathType Leaf) {
            Remove-Item -Path $Rm -Force
        } else {
            Remove-Item -Path $Rm -Recurse -Force
        }
        Write-Host "Delete $Rm"
    }
}

$CWD = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent
Set-Location -Path $CWD

# check cp files changes
$PathPairs = @(
    @("..\..\..\common\app\index_manager\core\blob_io.py", ".\core\blob_io.py"),
    @("..\..\..\common\app\index_manager\core\queue_io.py", ".\core\queue_io.py"),
    @("..\..\..\common\app\index_manager\core\util.py", ".\core\util.py"),
    @("..\..\..\common\app\index_manager\app.py", ".\app.py")
)
Compare-PathPairs $PathPairs

# rm build file
$Paths = @(
    ".\.vscode",
    ".\core",
    ".\app.py",
    ".\Dockerfile",
    ".\requirements.txt"
)
Remove-Paths $Paths