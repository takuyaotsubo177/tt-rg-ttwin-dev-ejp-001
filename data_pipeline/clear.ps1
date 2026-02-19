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
    @("..\..\common\data_pipeline\src\function_app.py", ".\src\function_app.py"),
    @("..\..\common\data_pipeline\src\error.py", ".\src\error.py"),
    @("..\..\common\data_pipeline\src\indexer.py", ".\src\indexer.py"),
    @("..\..\common\data_pipeline\src\loader.py", ".\src\loader.py"),
    @("..\..\common\data_pipeline\src\pipeline.py", ".\src\pipeline.py"),
    @("..\..\common\data_pipeline\src\queue_io.py", ".\src\queue_io.py"),
    @("..\..\common\data_pipeline\src\util.py", ".\src\util.py")
)
Compare-PathPairs $PathPairs

# rm build file
$Paths = @(
    ".\.vscode",
    ".\src\core",
    ".\src\function_app.py",
    ".\src\host.json",
    ".\src\requirements.txt",
    ".\.funcignore"
)
Remove-Paths $Paths