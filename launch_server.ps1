$env:FLASK_APP = "api"
$env:FLASK_ENV = "development"
cd labelprop
flask run --host 0.0.0.0 --port 5000

