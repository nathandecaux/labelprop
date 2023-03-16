#Catch arguments -p or --port and -h or --host
$port = 6000
$ip = "0.0.0.0"
$portIndex = $args.IndexOf("-p")
if ($portIndex -ne -1) {
    $port = $args[$portIndex + 1]
}
$hostIndex = $args.IndexOf("-h")
if ($hostIndex -ne -1) {
    $ip = $args[$hostIndex + 1]
}

$env:FLASK_APP = "api"
$env:FLASK_ENV = "development"
cd labelprop
flask run --host $ip --port $port 