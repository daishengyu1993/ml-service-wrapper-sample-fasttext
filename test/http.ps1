
$base_url = "http://localhost:5000"

Push-Location $PSScriptRoot\..;
Try {

    $input_data = Import-Csv -Path ".\data\input\Data.csv"

    $body = @{
        parameters = @{
            IdFieldName = "Id"
        };
        inputs = @{
            Data = $input_data
        }
    }

    $body_json = $body | ConvertTo-Json -Depth 100

    $predict_resp = Invoke-WebRequest `
        -Method "POST" `
        -ContentType "application/json; charset=utf-8" `
        -Body $body_json `
        $base_url/api/process/batch

    $response = $predict_resp.Content | ConvertFrom-Json

    $results = $response.outputs.Results | ForEach-Object { [PSCustomObject]$_ }

    $results | Export-Csv -Path ".\data\output\Results.csv" -Delimiter ',' -NoTypeInformation
}
Finally {
    Pop-Location
}
